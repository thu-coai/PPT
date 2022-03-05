# coding=utf-8


"""Utilities for logging and serialization"""

import os
import random
import numpy as np
import torch
import time

from fp16 import FP16_Optimizer
import mpu
import deepspeed
from apex.optimizers import FusedAdam as Adam
from fp16 import FP16_Module
from fp16 import FP16_Optimizer
from learning_rates import AnnealingLR
from model import EncDecModel, EncDecConfig
from model import (
    enc_dec_get_params_for_weight_decay_optimization,
    enc_dec_get_params_for_prompt_optimization,
)

from model import DistributedDataParallel as DDP

def print_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def print_args(args):
    """Print arguments."""

    print('arguments:', flush=True)
    for arg in vars(args):
        dots = '.' * (29 - len(arg))
        print('  {} {} {}'.format(arg, dots, getattr(args, arg)), flush=True)


def save_rank_0(args, message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            with open(args.log_file, "a") as f:
                f.write(message + "\n")
                f.flush()
    else:
        with open(args.log_file, "a") as f:
            f.write(message + "\n")
            f.flush()


def get_model(args, vocab_size, prompt_config=None):
    """Build the model."""

    print_rank_0('building Enc-Dec model ...')
    config = EncDecConfig.from_json_file(args.model_config)
    config.vocab_size = vocab_size
    model = EncDecModel(config,
                        parallel_output=True,
                        checkpoint_activations=args.checkpoint_activations,
                        checkpoint_num_layers=args.checkpoint_num_layers,
                        prompt_config=prompt_config)

    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if args.deepspeed and args.fp16:
        model.half()

    # GPU allocation.
    model.cuda(torch.cuda.current_device())
    if args.prompt_tune and prompt_config["init_scratch"]:
        model.init_prompt_embeds()

    # Fp16 conversion.
    if args.fp16:
        model = FP16_Module(model)

    # Wrap model for distributed training.
    model = DDP(model)

    return model


def get_optimizer(model, args, prompt_config=None):
    """Set up the optimizer."""

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, (DDP, FP16_Module)):
        model = model.module
    if args.prompt_tune and prompt_config["fix_model"]:
        param_groups = enc_dec_get_params_for_prompt_optimization(model)
    else:
        param_groups = enc_dec_get_params_for_weight_decay_optimization(model)
    
    # Add model parallel attribute if it is not set.
    for param_group in param_groups:
        for param in param_group['params']:
            if not hasattr(param, 'model_parallel'):
                param.model_parallel = False

    if args.cpu_optimizer:
        if args.cpu_torch_adam:
            cpu_adam_optimizer = torch.optim.Adam
        else:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            cpu_adam_optimizer = DeepSpeedCPUAdam
        optimizer = cpu_adam_optimizer(param_groups,
                        lr=args.lr, weight_decay=args.weight_decay)
    else:
        # Use FusedAdam.
        optimizer = Adam(param_groups,
                         lr=args.lr, weight_decay=args.weight_decay)

    print(f'Optimizer = {optimizer.__class__.__name__}')
    if args.deepspeed:
        # fp16 wrapper is not required for DeepSpeed.
        return optimizer

    # Wrap into fp16 optimizer.
    if args.fp16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale,
                                   dynamic_loss_args={
                                       'scale_window': args.loss_scale_window,
                                       'min_scale': args.min_scale,
                                       'delayed_shift': args.hysteresis})

    if torch.distributed.get_rank() == 0:
        print(optimizer.param_groups)

    return optimizer


def get_learning_rate_scheduler(optimizer, args):
    """Build the learning rate scheduler."""

    # Add linear learning rate scheduler.
    if args.lr_decay_iters is not None:
        num_iters = args.lr_decay_iters
    else:
        num_iters = args.train_iters
    num_iters = max(1, num_iters)
    init_step = -1
    if args.warmup_iter > 0:
        warmup_iter = args.warmup_iter
    else:
        warmup_iter = args.warmup * num_iters
    lr_scheduler = AnnealingLR(optimizer,
                               start_lr=args.lr,
                               warmup_iter=warmup_iter,
                               num_iters=num_iters,
                               decay_style=args.lr_decay_style,
                               last_iter=init_step,
                               gradient_accumulation_steps=args.gradient_accumulation_steps)

    return lr_scheduler


def setup_model_and_optimizer(args, vocab_size, ds_config, prompt_config=None):
    """Setup model and optimizer."""

    model = get_model(args, vocab_size, prompt_config)
    optimizer = get_optimizer(model, args, prompt_config)
    lr_scheduler = get_learning_rate_scheduler(optimizer, args)

    if args.deepspeed:
        print_rank_0("DeepSpeed is enabled.")

        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            args=args,
            lr_scheduler=lr_scheduler,
            mpu=mpu,
            dist_init_required=False,
            config_params=ds_config
        )

    print(args.load)
    if args.load is not None:
        args.iteration = load_checkpoint(model, optimizer, lr_scheduler, args, prompt_config)
    else:
        args.iteration = 0

    return model, optimizer, lr_scheduler


def set_deepspeed_activation_checkpointing(args):

    deepspeed.checkpointing.configure(mpu, deepspeed_config=args.deepspeed_config, num_checkpoints=args.num_checkpoints)
    mpu.checkpoint = deepspeed.checkpointing.checkpoint
    mpu.get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
    mpu.model_parallel_cuda_manual_seed = deepspeed.checkpointing.model_parallel_cuda_manual_seed


def initialize_distributed(args):
    """Initialize torch.distributed."""

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)
    # Call the init process
    deepspeed.init_distributed()

    # Set the model-parallel / data-parallel communicators.
    mpu.initialize_model_parallel(args.model_parallel_size)

    # Optional DeepSpeed Activation Checkpointing Features
    if args.deepspeed and args.deepspeed_activation_checkpointing:
        set_deepspeed_activation_checkpointing(args)


def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        mpu.model_parallel_cuda_manual_seed(seed)


def save_checkpoint(iteration, model, optimizer,
                    lr_scheduler, args, save_dir=None):
    """Save a model checkpoint."""
    save_ds_checkpoint(iteration, model, args, save_dir)

    # Wait so everyone is done (necessary)
    torch.distributed.barrier()
    # And update the latest iteration
    if torch.distributed.get_rank() == 0:
        tracker_filename = os.path.join(args.save, 'latest_checkpointed_iteration.txt')
        with open(tracker_filename, 'w') as f:
            f.write(str(iteration))
    # Wait so everyone is done (not necessary)
    torch.distributed.barrier()


def save_ds_checkpoint(iteration, model, args, save_dir=None):
    """Save a model checkpoint."""

    sd = {}
    sd['iteration'] = iteration

    if args.save_prompt_only:
        prompt = model.module.module.module.get_prompt_embeds()
        save_prompt(args.save if save_dir is None else save_dir, iteration, prompt["encoder"])
    else:
        model.save_checkpoint(args.save if save_dir is None else save_dir, str(iteration), client_state = sd, save_zero=False)


def save_prompt(save_dir, iteration, prompt_embeds):
    save_path = os.path.join(save_dir, "prompt-{}.pt".format(iteration))
    if torch.distributed.get_rank() == 0:
        torch.save(prompt_embeds, save_path)


def get_checkpoint_iteration(args):
    # Read the tracker file and set the iteration.
    tracker_filename = os.path.join(args.load, 'latest_checkpointed_iteration.txt')
    if not os.path.isfile(tracker_filename):
        print_rank_0('WARNING: could not find the metadata file {} '.format(
            tracker_filename))
        print_rank_0('    will not load any checkpoints and will start from '
                     'random')
        return 0, False, False
    iteration = 0
    release = False
    with open(tracker_filename, 'r') as f:
        metastring = f.read().strip()
        try:
            iteration = int(metastring)
        except ValueError:
            release = metastring == 'release'
            if not release:
                print_rank_0('ERROR: Invalid metadata file {}. Exiting'.format(
                    tracker_filename))
                exit()

    assert iteration > 0 or release, 'error parsing metadata file {}'.format(
        tracker_filename)
    
    return iteration, release, True


def load_prompt(load_dir):
    prompt = torch.load(load_dir, map_location=lambda storage, loc: storage)
    return prompt


def load_checkpoint(model, optimizer, lr_scheduler, args, prompt_config=None):
    """Load a model checkpoint."""

    iteration, release, success = get_checkpoint_iteration(args)

    if not success:
        return 0

    mp_rank = mpu.get_model_parallel_rank()
    checkpoint_name = os.path.join(args.load,
                        str(iteration),
                        'mp_rank_{:02d}'.format(mp_rank) + '_model_states.pt')

    if not os.path.exists(checkpoint_name):
        print('Client provided checkpoint load path: {} does not exist ... skip checkpoint load'.format(checkpoint_name))
        if mpu.get_data_parallel_rank() == 0:
            print("Unable to load checkpoint.")
        return iteration

    print('loading checkpoint: {}'.format(checkpoint_name))
    sd = torch.load(checkpoint_name, map_location=lambda storage, loc: storage)

    if args.prompt_tune:
        load_prompt_path = prompt_config.get("load_prompt")
        if load_prompt_path is not None and len(load_prompt_path) > 0:
            prompt_embeds = load_prompt(load_prompt_path)
            sd["module"]["encoder.prompt_embeds.weight"] = prompt_embeds                

    model.module.load_state_dict(sd["module"], strict=False)

    iteration = sd['iteration']

    torch.distributed.barrier()
    if mpu.get_data_parallel_rank() == 0:
        print('  successfully loaded {}'.format(checkpoint_name))

    return iteration


class Timers:
    """Group of timers."""

    class Timer:
        """Timer."""

        def __init__(self, name):
            self.name_ = name
            self.elapsed_ = 0.0
            self.started_ = False
            self.start_time = time.time()

        def start(self):
            """Start the timer."""
            assert not self.started_, 'timer has already been started'
            torch.cuda.synchronize()
            self.start_time = time.time()
            self.started_ = True

        def stop(self):
            """Stop the timer."""
            assert self.started_, 'timer is not started'
            torch.cuda.synchronize()
            self.elapsed_ += (time.time() - self.start_time)
            self.started_ = False

        def reset(self):
            """Reset timer."""
            self.elapsed_ = 0.0
            self.started_ = False

        def elapsed(self, reset=True):
            """Calculate the elapsed time."""
            started_ = self.started_
            # If the timing in progress, end it first.
            if self.started_:
                self.stop()
            # Get the elapsed time.
            elapsed_ = self.elapsed_
            # Reset the elapsed time
            if reset:
                self.reset()
            # If timing was in progress, set it back.
            if started_:
                self.start()
            return elapsed_

    def __init__(self):
        self.timers = {}

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = self.Timer(name)
        return self.timers[name]

    def log(self, names, normalizer=1.0, reset=True):
        """Log a group of timers."""
        assert normalizer > 0.0
        string = 'time (ms)'
        for name in names:
            elapsed_time = self.timers[name].elapsed(
                reset=reset) * 1000.0 / normalizer
            string += ' | {}: {:.2f}'.format(name, elapsed_time)
        print_rank_0(string)
