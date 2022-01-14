# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for logging and serialization"""

USE_TORCH_DDP = False

import os
import random
import time
import numpy as np
import torch

from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from fp16 import FP16_Optimizer
import mpu
import deepspeed
from apex.optimizers import FusedAdam as Adam
from fp16 import FP16_Module
from fp16 import FP16_Optimizer
from learning_rates import AnnealingLR
from model import EncDecModel, EncDecConfig
from model import enc_dec_get_params_for_weight_decay_optimization, enc_dec_get_params_for_prompt_optimization, enc_dec_get_params_for_optimization_wo_prompt

if USE_TORCH_DDP:
    from torch.nn.parallel.distributed import DistributedDataParallel as DDP
else:
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
                        data_hack="chid" if args.data_name == "chid3" else None,
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
    if USE_TORCH_DDP:
        i = torch.cuda.current_device()
        model = DDP(model, device_ids=[i], output_device=i,
                    process_group=mpu.get_data_parallel_group())
    else:
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
        if args.prompt_tune and prompt_config.get("fix_prompt", False):
            param_groups = enc_dec_get_params_for_optimization_wo_prompt(model)
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

    if args.prompt_tune:
        if prompt_config["init_from_vocab"]:
            model.module.module.module.init_prompt_embeds()
        # load_prompt_path = prompt_config.get("load_prompt")
        # if load_prompt_path is not None and len(load_prompt_path) > 0:
        #     prompt_embeds = torch.load(load_prompt_path)
        #     model.module.module.module.load_prompt_embeds(prompt_embeds)

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
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port
    deepspeed.init_distributed()

    # Set the model-parallel / data-parallel communicators.
    mpu.initialize_model_parallel(args.model_parallel_size)

    # Optional DeepSpeed Activation Checkpointing Features
    #
    if args.deepspeed and args.deepspeed_activation_checkpointing:
        set_deepspeed_activation_checkpointing(args)


def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        mpu.model_parallel_cuda_manual_seed(seed)


def get_checkpoint_name(checkpoints_path, iteration, release=False, zero=False):
    if release:
        d = 'release'
    else:
        d = 'iter_{:07d}'.format(iteration)
    if zero:
        dp_rank = mpu.get_data_parallel_rank()
        d += '_zero_dp_rank_{}'.format(dp_rank)
    return os.path.join(checkpoints_path, d,
                        'mp_rank_{:02d}'.format(mpu.get_model_parallel_rank()),
                        'model_optim_rng.pt')


def ensure_directory_exists(filename):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_checkpoint_tracker_filename(checkpoints_path):
    return os.path.join(checkpoints_path, 'latest_checkpointed_iteration.txt')


def save_zero_checkpoint(args, iteration, optimizer):
    zero_sd = {'iteration': iteration,
               'optimizer_state_dict': optimizer.state_dict()}
    zero_checkpoint_name = get_checkpoint_name(args.save, iteration, zero=True)
    ensure_directory_exists(zero_checkpoint_name)
    torch.save(zero_sd, zero_checkpoint_name)
    print('  successfully saved {}'.format(zero_checkpoint_name))


def save_checkpoint(iteration, model, optimizer,
                    lr_scheduler, args, save_dir=None):
    """Save a model checkpoint."""
    if args.deepspeed:
        save_ds_checkpoint(iteration, model, args, save_dir)
    else:
        # Only rank zer0 of the data parallel writes to the disk.
        if isinstance(model, torchDDP):
            model = model.module

        if mpu.get_data_parallel_rank() == 0:
            checkpoint_name = get_checkpoint_name(args.save, iteration)
            print('global rank {} is saving checkpoint at iteration {:7d} to {}'.
                format(torch.distributed.get_rank(), iteration, checkpoint_name))

            sd = {}
            sd['iteration'] = iteration
            sd['model'] = model.state_dict()

            # Optimizer stuff.
            if not args.no_save_optim:
                if optimizer is not None:
                    sd['optimizer'] = optimizer.state_dict()
                if lr_scheduler is not None:
                    sd['lr_scheduler'] = lr_scheduler.state_dict()

            # rng states.
            if not args.no_save_rng:
                sd['random_rng_state'] = random.getstate()
                sd['np_rng_state'] = np.random.get_state()
                sd['torch_rng_state'] = torch.get_rng_state()
                sd['cuda_rng_state'] = torch.cuda.get_rng_state()
                sd['rng_tracker_states'] = mpu.get_cuda_rng_tracker().get_states()


            ensure_directory_exists(checkpoint_name)
            torch.save(sd, checkpoint_name)
            print('  successfully saved {}'.format(checkpoint_name))

    # Wait so everyone is done (necessary)
    torch.distributed.barrier()
    # And update the latest iteration
    if torch.distributed.get_rank() == 0:
        tracker_filename = get_checkpoint_tracker_filename(args.save)
        with open(tracker_filename, 'w') as f:
            f.write(str(iteration))
    # Wait so everyone is done (not necessary)
    torch.distributed.barrier()


def save_ds_checkpoint(iteration, model, args, save_dir=None):
    """Save a model checkpoint."""

    sd = {}
    sd['iteration'] = iteration
    # rng states.
    if not args.no_save_rng:
        sd['random_rng_state'] = random.getstate()
        sd['np_rng_state'] = np.random.get_state()
        sd['torch_rng_state'] = torch.get_rng_state()
        sd['cuda_rng_state'] = torch.cuda.get_rng_state()
        sd['rng_tracker_states'] = mpu.get_cuda_rng_tracker().get_states()

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
    tracker_filename = get_checkpoint_tracker_filename(args.load)
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


def load_prompt(args):
    prompt = torch.load(args.load_prompt)
    return prompt


def load_checkpoint(model, optimizer, lr_scheduler, args, prompt_config=None):
    """Load a model checkpoint."""

    iteration, release, success = get_checkpoint_iteration(args)

    if not success:
        return 0
        
    if args.deepspeed:

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
                prompt_embeds = torch.load(load_prompt_path, map_location=lambda storage, loc: storage)
                sd["module"]["encoder.prompt_embeds.weight"] = prompt_embeds                

        model.module.load_state_dict(sd["module"], strict=False)

        # checkpoint_name, sd = model.load_checkpoint(
        #     args.load,
        #     iteration,
        #     load_module_strict=False,
        #     load_optimizer_states=args.load_oprimizer_states,
        #     load_lr_scheduler_states=args.load_lr_scheduler_states)

        # if checkpoint_name is None:
        #     if mpu.get_data_parallel_rank() == 0:
        #         print("Unable to load checkpoint.")
        #     return iteration

    else:
        
        # Checkpoint.
        checkpoint_name = get_checkpoint_name(args.load, iteration, release)
        
        if mpu.get_data_parallel_rank() == 0:
            print('global rank {} is loading checkpoint {}'.format(
                torch.distributed.get_rank(), checkpoint_name))

        # Load the checkpoint.
        sd = torch.load(checkpoint_name, map_location='cpu')

        if isinstance(model, torchDDP):
            model = model.module
        
        # Model.
        try:
            model.load_state_dict(sd['model'])
        except KeyError:
            print_rank_0('A metadata file exists but unable to load model '
                        'from checkpoint {}, exiting'.format(checkpoint_name))
            exit()

        # Optimizer.
        if not release and not args.finetune and not args.no_load_optim:
            try:
                if optimizer is not None:
                    optimizer.load_state_dict(sd['optimizer'])
                if lr_scheduler is not None:
                    lr_scheduler.load_state_dict(sd['lr_scheduler'])
            except KeyError:
                print_rank_0('Unable to load optimizer from checkpoint {}, exiting. '
                            'Specify --no-load-optim or --finetune to prevent '
                            'attempting to load the optimizer '
                            'state.'.format(checkpoint_name))
                exit()

    # Iterations.
    if args.finetune or release:
        iteration = 0
    else:
        try:
            iteration = sd['iteration']
        except KeyError:
            try: # Backward compatible with older checkpoints
                iteration = sd['total_iters']
            except KeyError:
                print_rank_0('A metadata file exists but Unable to load iteration '
                             ' from checkpoint {}, exiting'.format(checkpoint_name))
                exit()
                
    # rng states.
    if not release and not args.finetune and not args.no_load_rng:
        try:
            random.setstate(sd['random_rng_state'])
            np.random.set_state(sd['np_rng_state'])
            torch.set_rng_state(sd['torch_rng_state'])
            torch.cuda.set_rng_state(sd['cuda_rng_state'])
            mpu.get_cuda_rng_tracker().set_states(sd['rng_tracker_states'])
        except KeyError:
            print_rank_0('Unable to load optimizer from checkpoint {}, exiting. '
                         'Specify --no-load-optim or --finetune to prevent '
                         'attempting to load the optimizer '
                         'state.'.format(checkpoint_name))
            exit()

    torch.distributed.barrier()
    if mpu.get_data_parallel_rank() == 0:
        print('  successfully loaded {}'.format(checkpoint_name))

    return iteration
