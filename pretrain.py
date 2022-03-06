from datetime import datetime
import os
import math
import torch
import json
from tqdm import tqdm

from arguments import get_args
from tokenization_t5 import EncDecTokenizer
from fp16 import FP16_Module
from model import EncDecModel, EncDecConfig


from model import DistributedDataParallel as DDP
import mpu
from utils import save_checkpoint
from utils import print_args
from utils import print_rank_0, save_rank_0
from utils import setup_model_and_optimizer, set_random_seed, initialize_distributed
from utils import Timers
import torch.distributed as dist

from data_utils import *

from samplers import DistributedBatchSampler


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

    model = DDP(model)

    return model


def get_masks_and_position_ids(args,
                               tokenizer: EncDecTokenizer,
                               contexts,
                               targets,
                               labels):
    # Extract batch size and sequence length.
    batch_size, enc_seq_length = contexts.size()

    # Enc Attention mask.
    enc_attn_mask = torch.zeros(
        batch_size, 1, enc_seq_length, enc_seq_length, device=contexts.device)

    for mask, context in zip(enc_attn_mask, contexts):
        l = (context != tokenizer.pad_id).long().sum().data
        mask[0][:l, :l] = 1.0

    # Enc Position ids.
    enc_pos_ids = torch.arange(enc_seq_length, dtype=torch.long, device=contexts.device)
    enc_pos_ids = enc_pos_ids.unsqueeze(0).expand_as(contexts)
    # We need to clone as the ids will be modifed based on batch index.
    
    batch_size, dec_seq_length = targets.size()
    # Dec Attention mask
    dec_attn_mask = torch.zeros(batch_size, 1, dec_seq_length, dec_seq_length, device=targets.device)
    for mask, target in zip(dec_attn_mask, targets):
        l = (target != tokenizer.pad_id).long().sum().data
        mask[0][:l, :l] = torch.tril(torch.ones(l, l, device=targets.device))

    # Dec Position ids.
    dec_pos_ids = torch.arange(dec_seq_length, dtype=torch.long, device=targets.device)
    dec_pos_ids = dec_pos_ids.unsqueeze(0).expand_as(targets)
    # We need to clone as the ids will be modifed based on batch index.

    # Loss mask.
    loss_mask = torch.ones(targets.size(), dtype=torch.float, device=targets.device)
    loss_mask[labels == tokenizer.pad_id] = 0.0

    # Cross Attention Mask
    cross_attn_mask = torch.zeros(batch_size, 1, dec_seq_length, enc_seq_length, device=contexts.device)
    for mask, context, target in zip(cross_attn_mask, contexts, targets):
        l_e = (context != tokenizer.pad_id).long().sum().data
        l_d = (target != tokenizer.pad_id).long().sum().data
        mask[0][:l_d, :l_e] = 1.0

    if args.fp16:
        enc_attn_mask = enc_attn_mask.half()
        dec_attn_mask = dec_attn_mask.half()
        cross_attn_mask = cross_attn_mask.half()

    model_batch = {
        "enc_attention_mask": enc_attn_mask,
        "enc_position_ids": enc_pos_ids,
        "dec_attention_mask": dec_attn_mask,
        "dec_position_ids": dec_pos_ids,
        "cross_attention_mask": cross_attn_mask,
    }

    no_model_batch = {
        "loss_mask": loss_mask
    }

    return model_batch, no_model_batch


def get_batch(tokenizer, data_iterator, args, timers):
    # Items and their type.
    datatype = torch.int64

    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    keys = [
        "contexts",
        "targets",
        "labels",
    ]

    # timers('data loader').stop()
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    contexts = data_b['contexts'].long()
    targets = data_b['targets'].long()
    labels = data_b['labels'].long()

    # Get the masks and postition ids.
    model_b, no_model_b = get_masks_and_position_ids(
        args,
        tokenizer,
        contexts,
        targets,
        labels)

    batch = {
        "enc_input_ids": contexts,
        "dec_input_ids": targets,
        **model_b
    }

    no_model_batch = {
        "labels": labels,
        **no_model_b
    }

    return batch, no_model_batch


def forward_step(tokenizer, data_iterator, model, args, timers):
    """Forward step."""

    # Get the batch.
    # timers('batch generator').start()
    # tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator, args, timers)
    batch, no_model_batch = get_batch(tokenizer, data_iterator, args, timers)
    # timers('batch generator').stop()
        
    # Forward model.
    output = model(**batch)
    logits = output["lm_logits"]
    losses = mpu.vocab_parallel_cross_entropy(logits.contiguous().float(), no_model_batch["labels"])
    loss_mask = no_model_batch["loss_mask"].view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    return loss


def backward_step(optimizer, model, lm_loss, args, timers):
    """Backward step."""
    # Total loss.

    loss = lm_loss

    # Backward pass.
    if args.deepspeed:
        model.backward(loss)
    else:
        optimizer.zero_grad()
        if args.fp16:
            optimizer.backward(loss, update_master_grads=False)
        else:
            loss.backward()

    # Reduce across processes.
    lm_loss_reduced = lm_loss

    reduced_losses = lm_loss.view(1)

    if args.deepspeed:
        # DeepSpeed backward propagation already addressed all reduce communication.
        # Reset the timer to avoid breaking timer logs below.
        timers('allreduce').reset()
    else:
        torch.distributed.all_reduce(reduced_losses.data)
        reduced_losses.data = reduced_losses.data / args.world_size

    lm_loss_reduced = reduced_losses

    # Update master gradients.
    if not args.deepspeed:
        if args.fp16:
            optimizer.update_master_grads()

        # Clipping gradients helps prevent the exploding gradient.
        if args.clip_grad > 0:
            if not args.fp16:
                mpu.clip_grad_norm(model.parameters(), args.clip_grad)
            else:
                optimizer.clip_master_grads(args.clip_grad)

    return lm_loss_reduced


def train_step(tokenizer, data_iterator, model, optimizer, lr_scheduler,
               args, timers):
    """Single training step."""

    lm_loss = forward_step(tokenizer, data_iterator, model, args, timers)
    lm_loss_reduced = backward_step(optimizer, model, lm_loss, args, timers)

    if dist.get_rank() == 0:
        print("loss", lm_loss_reduced)

    # Update parameters.
    skipped_iter = 0
    if args.deepspeed:
        model.step()
    else:
        optimizer.step()

        # Update learning rate.
        if not (args.fp16 and optimizer.overflow):
            lr_scheduler.step()
        else:
            skipped_iter = 1

    return lm_loss_reduced, skipped_iter


def train(tokenizer, model, optimizer, lr_scheduler,
          train_data_iterator, val_data_iterator, timers, args):
    """Train the model."""

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_lm_loss = 0.0

    # Iterations.
    skipped_iters = 0

    timers('interval time').start()
    for iteration in tqdm(range(args.iteration, args.train_iters), disable=(torch.distributed.get_rank() != 0), desc="Pretaining"):

        lm_loss, skipped_iter = train_step(tokenizer, train_data_iterator,
                                           model,
                                           optimizer,
                                           lr_scheduler,
                                           args, timers)
        skipped_iters += skipped_iter

        # Update losses.
        total_lm_loss += lm_loss.data.detach().float()

        # Logging.
        if iteration % args.log_interval == 0:
            learning_rate = optimizer.param_groups[0]['lr']
            avg_lm_loss = total_lm_loss.item() / args.log_interval
            elapsed_time = timers('interval time').elapsed()
            log_string = ' iteration {:8d}/{:8d} |'.format(iteration,
                                                            args.train_iters)
            log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
                elapsed_time * 1000.0 / args.log_interval)
            log_string += ' learning rate {:.3} |'.format(learning_rate)
            log_string += ' lm loss {:.6} |'.format(avg_lm_loss)
            if args.fp16:
                log_string += ' loss scale {:.1f} |'.format(
                    optimizer.cur_scale if args.deepspeed else optimizer.loss_scale)
            print_rank_0(log_string)
            save_rank_0(args, log_string)
            total_lm_loss = 0.0

        # Checkpointing
        if args.save and args.save_interval and iteration % args.save_interval == 0:
            save_checkpoint(iteration, model, optimizer, lr_scheduler, args)

        # Evaluation
        if args.eval_interval and iteration % args.eval_interval == 0 and args.do_valid:
            prefix = 'iteration {}'.format(iteration)
            evaluate_and_print_results(
                tokenizer, prefix, val_data_iterator, model, args, timers, False)

    return iteration, skipped_iters


def evaluate(tokenizer, data_iterator, model, args, timers, verbose=False):
    """Evaluation."""

    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_lm_loss = 0

    with torch.no_grad():
        for iteration in tqdm(range(args.eval_iters), disable=(torch.distributed.get_rank() != 0), desc="Evaluating"):
            iteration += 1
            if verbose and iteration % args.log_interval == 0:
                msg = 'Evaluating iter {}/{}'.format(iteration, args.eval_iters)
                print_rank_0(msg)
                save_rank_0(args, msg)
            # Forward evaluation.
            lm_loss = forward_step(tokenizer, data_iterator, model, args, timers)

            # Reduce across processes.
            if isinstance(model, DDP):
                torch.distributed.all_reduce(lm_loss.data)
                lm_loss.data = lm_loss.data / args.world_size

            total_lm_loss += lm_loss.data.detach().float().item()

    # Move model back to the train mode.
    model.train()

    total_lm_loss /= args.eval_iters
    return total_lm_loss


def evaluate_and_print_results(tokenizer, prefix, data_iterator, model,
                               args, timers, verbose=False):
    """Helper function to evaluate and dump results on screen."""
    lm_loss = evaluate(tokenizer, data_iterator, model, args, timers, verbose)
    lm_ppl = math.exp(min(20, lm_loss))
    string = '-' * 100 + "\n"
    string += ' validation loss at {} | '.format(prefix)
    string += 'LM loss: {:.6} | '.format(lm_loss)
    string += 'LM PPL: {:.6}'.format(lm_ppl)
    length = len(string) + 1
    string = '-' * length + "\n" + string + "\n" + '-' * length
    print_rank_0(string)
    save_rank_0(args, string)

    return lm_loss


def make_data_loader(dataset):
    """Buld dataloader given an input dataset."""
    if dataset is None:
        return None
    args = get_args()

    # Data parallel arguments.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    global_batch_size = args.batch_size * world_size
    num_workers = args.num_workers

    # Use a simple sampler with distributed batch sampler.
    sampler = torch.utils.data.SequentialSampler(dataset)
    batch_sampler = DistributedBatchSampler(sampler=sampler,
                                            batch_size=global_batch_size,
                                            drop_last=True,
                                            rank=rank,
                                            world_size=world_size)
    # Torch dataloader.
    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=num_workers,
                                       pin_memory=True)


def main():
    """Main training program."""

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Timer.
    timers = Timers()

    # Arguments.
    args = get_args()

    os.makedirs(args.save, exist_ok=True)

    # Pytorch distributed.
    initialize_distributed(args)
    if torch.distributed.get_rank() == 0:
        print('Pretrain Enc-Dec model')
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f)

    # Random seeds for reproducability.
    set_random_seed(args.seed)
    device = torch.cuda.current_device()

    # setup tokenizer
    tokenizer = EncDecTokenizer(os.path.join(args.tokenizer_path, 'spiece.model'))
    
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size

    prompt_config = None
    if args.prompt_tune:
        with open(args.prompt_config, "r") as f:
            prompt_config = json.load(f)
            for t in ["enc", "dec"]:
                prompt_config[t]["init_ids"] = tokenizer.encode(prompt_config[t]["init_tokens"])
                pad_num = prompt_config[t]["prompt_len"] - len(prompt_config[t]["init_ids"])
                prompt_config[t]["init_ids"].extend(tokenizer.convert_tokens_to_ids([prompt_config[t]["default_init_token"] for _ in range(pad_num)]))
                prompt_config[t]["init_ids"] = torch.tensor(prompt_config[t]["init_ids"], dtype=torch.long).to(device)


    # Model, optimizer, and learning rate.
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args, tokenizer.vocab_size, ds_config, prompt_config)
    optimizer.cur_scale = 4096
    
    if torch.distributed.get_rank() == 0:
        print(args.iteration)
    
    train_data_iterator, val_data_iterator, test_data_iterator = \
            build_train_valid_test_data_iterators(
                    train_valid_test_dataset_provider, args, tokenizer, prompt_config)

    iteration = 0
    if args.train_iters > 0:
        iteration, skipped = train(tokenizer, model, optimizer,
                                   lr_scheduler,
                                   train_data_iterator,
                                   val_data_iterator,
                                   timers, args)

        prefix = 'the end of training for val data'
        evaluate_and_print_results(tokenizer, prefix, val_data_iterator,
                                                  model, args, False)

    if args.save and iteration != 0:
        save_checkpoint(iteration, model, optimizer, lr_scheduler, args)


def train_valid_test_dataset_provider(tokenizer, train_val_test_num_samples, prompt_config):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for Enc-Dec ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        tokenizer=tokenizer,
        data_class=DATA_CONFIG[args.pretrain_task]["dataset"],
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        enc_seq_length=args.enc_seq_length,
        dec_seq_length=args.dec_seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        prompt_config=prompt_config)
    print_rank_0("> finished creating Enc-Dec datasets ...")

    return train_ds, valid_ds, test_ds


def build_train_valid_test_data_iterators(
        build_train_valid_test_datasets_provider, args, tokenizer, prompt_config):
    """XXX"""

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0('> building train, validation, and test datasets ...')
    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_model_parallel_rank() == 0:
        # Rank, size, and global batch size.
        data_parallel_size = mpu.get_data_parallel_world_size()
        global_batch_size = args.batch_size * data_parallel_size

        # Number of train/valid/test samples.
        train_iters = args.train_iters
        eval_iters = (train_iters // args.eval_interval + 1) * args.eval_iters
        test_iters = args.eval_iters
        train_val_test_num_samples = [train_iters * global_batch_size,
                                      eval_iters * global_batch_size,
                                      test_iters * global_batch_size]
        print_rank_0(' > datasets target sizes (minimum size):')
        print_rank_0('    train:      {}'.format(train_val_test_num_samples[0]))
        print_rank_0('    validation: {}'.format(train_val_test_num_samples[1]))
        print_rank_0('    test:       {}'.format(train_val_test_num_samples[2]))

        # Build the datasets.
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets_provider(
            tokenizer, train_val_test_num_samples, prompt_config)

        # Build dataloders.
        train_dataloader = make_data_loader(train_ds)
        valid_dataloader = make_data_loader(valid_ds)
        test_dataloader = make_data_loader(test_ds)

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and args.train_iters > 0
        do_valid = valid_dataloader is not None and args.eval_iters > 0
        do_test = test_dataloader is not None and args.eval_iters > 0
        # Need to broadcast num_tokens and num_type_tokens.
        flags = torch.cuda.LongTensor(
            [int(do_train), int(do_valid), int(do_test)])
    else:
        flags = torch.cuda.LongTensor([0, 0, 0])

    # Broadcast num tokens.
    torch.distributed.broadcast(flags,
                                mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    args.do_train = flags[0].item()
    args.do_valid = flags[1].item()
    args.do_test = flags[2].item()

    # Shift the start iterations.
    if train_dataloader is not None:
        train_dataloader.batch_sampler.start_iter = args.iteration % \
            len(train_dataloader)
        print_rank_0('setting training data start iteration to {}'.
                     format(train_dataloader.batch_sampler.start_iter))
    if valid_dataloader is not None:
        start_iter_val = (args.iteration // args.eval_interval) * \
            args.eval_iters
        valid_dataloader.batch_sampler.start_iter = start_iter_val % \
            len(valid_dataloader)
        print_rank_0('setting validation data start iteration to {}'.
                     format(valid_dataloader.batch_sampler.start_iter))

    # Build iterators.
    if train_dataloader is not None:
        train_data_iterator = iter(train_dataloader)
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        valid_data_iterator = iter(valid_dataloader)
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        test_data_iterator = iter(test_dataloader)
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator


if __name__ == "__main__":
    main()
