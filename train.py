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

"""Pretrain Enc-Dec"""

# Flag to use Pytorch ddp which uses overlapping communication and computation.
USE_TORCH_DDP = False

import os
import torch
import json
import shutil
from sklearn.metrics import f1_score

from arguments import get_args
from tokenization_t5 import EncDecTokenizer

import mpu
from utils import save_checkpoint
from utils import print_args
from utils import print_rank_0, save_rank_0
from utils import setup_model_and_optimizer, set_random_seed, initialize_distributed

from samplers import DistributedBatchSampler, RandomSampler
from data_utils import *

from torch.utils.data import DataLoader, SequentialSampler


def forward_step(args, model_batch, no_model_batch, model, device, keep_enc_hidden=False, do_infer=False):
    for k in model_batch:
        model_batch[k] = model_batch[k].to(device)
    for k in no_model_batch:
        no_model_batch[k] = no_model_batch[k].to(device)

    if keep_enc_hidden:
        enc_outputs = model(**model_batch, only_encoder=True)
        enc_hidden_states = enc_outputs["encoder_last_hidden_state"]
        output = model(**model_batch, enc_hidden_states=enc_hidden_states)
    else:
        output = model(**model_batch)
    
    logits = output["lm_logits"]
    forw_out = {
        "logits": logits
    }
    if keep_enc_hidden:
        forw_out["enc_hidden_states"] = enc_hidden_states
    
    if not do_infer:
        losses = mpu.vocab_parallel_cross_entropy(logits.contiguous().float(), no_model_batch["labels"])

        loss_mask = no_model_batch["loss_mask"]
        # if torch.distributed.get_rank() == 0:
        #     print(losses)
        # exit(0)
        losses = (losses * loss_mask).sum(-1) / loss_mask.sum(-1)
        loss = losses.mean()

        forw_out["loss"] = loss
        forw_out["loss_batch"] = losses
    
    return forw_out


def backward_step(args, loss, model, optimizer):
    # backward
    if args.deepspeed:
        model.backward(loss)
    else:
        optimizer.zero_grad()
        if args.fp16:
            optimizer.backward(loss, update_master_grads=False)
        else:
            loss.backward()

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


def train(args, data_config, tokenizer, model, optimizer, lr_scheduler,
          train_dataset, train_dataloader, dev_dataset, dev_dataloader, eval_dataset, eval_dataloader, device, random_sampler: RandomSampler, prompt_config):
    """Train the model."""

    eval_func = data_config[args.data_name]["eval_func"]

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_loss = 0.0

    step, global_step = 1, 1

    best_accs = []

    for e in range(args.epochs):
        model.train()
        random_sampler.set_epoch(e)
        for model_batch, no_model_batch in train_dataloader:

            forw_out = forward_step(args, model_batch, no_model_batch, model, device)
            loss = forw_out["loss"]
            
            if torch.distributed.get_rank() == 0:
                print(loss)

            backward_step(args, loss, model, optimizer)

            # Update losses.
            total_loss += loss.item()

            if args.deepspeed:
                model.step()
            else:
                optimizer.step()
                if not (args.fp16 and optimizer.overflow):
                    lr_scheduler.step()

            # Logging.
            if global_step % args.log_interval == 0 and step % args.gradient_accumulation_steps == 0:
                learning_rate = optimizer.param_groups[0]['lr']
                avg_lm_loss = total_loss / (args.log_interval * args.gradient_accumulation_steps)
                log_string = 'epoch {:3d}/{:3d} |'.format(e, args.epochs)
                log_string += ' global iteration {:8d}/{:8d} |'.format(global_step, args.train_iters)
                log_string += ' learning rate {:.3} |'.format(learning_rate)
                log_string += ' lm loss {:.6} |'.format(avg_lm_loss)
                if args.fp16:
                    log_string += ' loss scale {:.1f} |'.format(optimizer.cur_scale if args.deepspeed else optimizer.loss_scale)
                print_rank_0(log_string)
                save_rank_0(args, log_string)
                total_loss = 0.0

            # Checkpointing
            if args.save and args.save_interval and global_step % args.save_interval == 0 and step % args.gradient_accumulation_steps == 0:
                save_checkpoint(global_step, model, optimizer, lr_scheduler, args)

            # Evaluation
            if args.eval_interval and global_step % args.eval_interval == 0 and step % args.gradient_accumulation_steps == 0 and args.do_valid:
                prefix = 'iteration {} | '.format(global_step)
                dev_loss, dev_acc = eval_func(args, tokenizer, data_config, dev_dataset, dev_dataloader, model, device, prompt_config, mode="dev")
                eval_loss, eval_acc = eval_func(args, tokenizer, data_config, eval_dataset, eval_dataloader, model, device, prompt_config, mode="test")

                model.train()
                log_string = prefix + " dev_loss: " + str(dev_loss) + " | dev acc(mrr, f1): " + str(dev_acc) 
                log_string = log_string + " | eval_loss: " + str(eval_loss) + " | eval acc(mrr, f1): " + str(eval_acc)
                print_rank_0(log_string)
                save_rank_0(args, log_string)

                if args.max_save > 0:
                    i = 0
                    while i < len(best_accs):
                        if best_accs[i][1] < dev_acc:
                            break
                        i += 1
                    if len(best_accs) < args.max_save or i < len(best_accs):
                        best_accs.insert(i, (global_step, dev_acc))
                        if len(best_accs) > args.max_save:
                            step_to_be_rm, acc_to_be_rm = best_accs[-1]
                            if torch.distributed.get_rank() == 0:
                                shutil.rmtree(os.path.join(args.save, "acc_{}_{:.3}".format(step_to_be_rm, acc_to_be_rm)))
                        save_checkpoint(global_step, model, optimizer, lr_scheduler, args, save_dir=os.path.join(args.save, "acc_{}_{:.3}".format(global_step, dev_acc)))
                        best_accs = best_accs[:args.max_save]

            step += 1
            if step % args.gradient_accumulation_steps == 0:
                global_step += 1

    return global_step


def evaluate(args, tokenizer: EncDecTokenizer, data_config, eval_dataset: EncDecDataset, eval_data_loader, model, device, prompt_config, mode='dev'):
    """Evaluation."""

    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_loss = 0.0
    step = 0

    all_idx = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for model_batch, no_model_batch in eval_data_loader:

            forw_out = forward_step(args, model_batch, no_model_batch, model, device, do_infer=(mode=="infer"))
            loss = forw_out["loss"].item() if "loss" in forw_out else 0
            total_loss += loss

            logits_list = [torch.zeros_like(forw_out["logits"]) for _ in range(mpu.get_model_parallel_world_size())]
            torch.distributed.all_gather(logits_list, forw_out["logits"], mpu.get_model_parallel_group())

            gathered_logits = torch.cat(logits_list, dim=-1)

            if args.from_lm:
                pred_token_logits = gathered_logits[:, 0, :]
            else:
                pred_token_logits = gathered_logits[:, 1, :]

            preds = torch.argmax(pred_token_logits, dim=-1)

            gathered_preds = [torch.zeros_like(preds) for _ in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather(gathered_preds, preds.contiguous(), mpu.get_data_parallel_group())
            all_preds.extend(gathered_preds)
            
            gathered_idx = [torch.zeros_like(no_model_batch["idx"]) for _ in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather(gathered_idx, no_model_batch["idx"].contiguous(), mpu.get_data_parallel_group())
            all_idx.extend(gathered_idx)

            if args.from_lm:
                labels = no_model_batch["labels"][:, 0]
            else:
                labels = no_model_batch["labels"][:, 1]
            gathered_labels = [torch.zeros_like(labels) for _ in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather(gathered_labels, labels.contiguous(), mpu.get_data_parallel_group())
            all_labels.extend(gathered_labels)

            step += 1

    total_loss /= step

    all_idx = torch.cat(all_idx, dim=0).cpu().tolist()
    all_preds = torch.cat(all_preds, dim=0).cpu().tolist()
    all_labels = torch.cat(all_labels, dim=0).cpu().tolist()

    eval_metrc = data_config[args.data_name]["eval_metric"]
    res = eval_metrc(args, tokenizer, all_preds, all_labels)

    return total_loss, res


def acc_metric(args, tokenizer: EncDecTokenizer, all_preds, all_labels):
    acc = sum([int(p == l) for p, l in zip(all_preds, all_labels)]) / len(all_preds)
    
    with open(os.path.join(args.save, "{}.txt".format(acc)), "w") as f:
        for p, l in zip(all_preds, all_labels):
            f.write(str(p) + "\t\t" + str(l) + "\n")
            if isinstance(p, list):
                f.write(tokenizer.decode(p) + "\t\t" + tokenizer.decode(l) + "\n")
            f.write("\n")

    return acc


def acc_f1_metric(args, tokenizer: EncDecTokenizer, all_preds, all_labels):
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    acc = sum([int(p == l) for p, l in zip(all_preds, all_labels)]) / len(all_preds)

    with open(os.path.join(args.save, "{}.txt".format(f1_macro)), "w") as f:
        for p, l in zip(all_preds, all_labels):
            f.write(str(p) + "\t\t" + str(l) + "\n")
            if isinstance(p, list):
                f.write(tokenizer.decode(p) + "\t\t" + tokenizer.decode(l) + "\n")
            f.write("\n")

    return [acc, f1_macro]


def load_data(args, data_config, data_type, tokenizer, prompt_config=None, ratio=1, num=-1, drop_last=True, do_infer=False):
    data_path = os.path.join(args.data_path, data_type + args.data_ext)

    # Data parallel arguments.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
    if data_type == "train":
        global_batch_size = args.batch_size * world_size
    else:
        global_batch_size = args.eval_batch_size * world_size

    num_workers = args.num_workers

    dataset = data_config[args.data_name]["dataset"](
        args,
        tokenizer,
        data_path,
        data_type,
        ratio=ratio,
        num=num,
        prefix=args.data_prefix,
        cache_path=data_config[args.data_name]["cache_path"],
        do_infer=do_infer,
        prompt_config=prompt_config)

    if data_type == 'train':
        sampler = RandomSampler(dataset)
        sampler.set_seed(args.seed)
    else:
        sampler = SequentialSampler(dataset)
    batch_sampler = DistributedBatchSampler(sampler=sampler,
                                            batch_size=global_batch_size,
                                            drop_last=drop_last,
                                            rank=rank,
                                            world_size=world_size)

    data_loader = DataLoader(dataset,
                             batch_sampler=batch_sampler,
                             num_workers=num_workers,
                             pin_memory=True,
                             collate_fn=dataset.collate)

    # Torch dataloader.
    return data_loader, dataset, sampler


def main():
    """Main training program."""

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

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
            if args.load_prompt is not None:
                prompt_config["load_prompt"] = args.load_prompt
            for t in ["enc", "dec"]:
                prompt_config[t]["init_ids"] = tokenizer.encode(prompt_config[t]["init_tokens"])
                pad_num = prompt_config[t]["prompt_len"] - len(prompt_config[t]["init_ids"])
                prompt_config[t]["init_ids"].extend(tokenizer.convert_tokens_to_ids([prompt_config[t]["default_init_token"] for _ in range(pad_num)]))

    data_config = {
        "boolq": {
            "dataset": BoolQDataset,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "boolq_uni": {
            "dataset": BoolQDatasetUni,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "rte": {
            "dataset": RTEDataset,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "rte_uni": {
            "dataset": RTEDatasetUni,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "cb": {
            "dataset": CBDataset,
            "eval_func": evaluate,
            "eval_metric": acc_f1_metric,
            "cache_path": None,
        },
        "cb_uni": {
            "dataset": CBDatasetUni,
            "eval_func": evaluate,
            "eval_metric": acc_f1_metric,
            "cache_path": None,
        },
        "race": {
            "dataset": RACEDataset,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "race_uni": {
            "dataset": RACEDatasetUni,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "sst2": {
            "dataset": SST2Dataset,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "sst2_uni": {
            "dataset": SST2DatasetUni,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None
        },
        "sst5": {
            "dataset": SST5Dataset,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "sst5_uni": {
            "dataset": SST5DatasetUni,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
    }

    if args.do_train:
        train_dataloader, train_dataset, random_sampler = load_data(args, data_config, 'train', tokenizer, prompt_config, ratio=args.train_ratio, num=args.train_num)
        dev_dataloader, dev_dataset, _  = load_data(args, data_config, 'dev32', tokenizer, prompt_config, ratio=args.dev_ratio, num=args.dev_num)
        eval_dataloader, eval_dataset, _ = load_data(args, data_config, 'valid', tokenizer, prompt_config, ratio=args.test_ratio, num=args.test_num)
        if args.train_iters == -1:
            args.train_iters = len(train_dataset) * args.epochs // (mpu.get_data_parallel_world_size() * args.batch_size * args.gradient_accumulation_steps)
    else:
        args.train_iters = 10 # a magic number

    log_string = "Total train epochs {} | Total train iters {} | ".format(args.epochs, args.train_iters)
    print_rank_0(log_string)
    save_rank_0(args, log_string)

    # Model, optimizer, and learning rate.
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args, tokenizer.vocab_size, ds_config, prompt_config)

    if args.do_train:
        train(args, data_config, tokenizer, model, optimizer, lr_scheduler, train_dataset, train_dataloader, dev_dataset, dev_dataloader, eval_dataset, eval_dataloader, device, random_sampler, prompt_config)

    if args.do_eval:
        eval_dataloader, eval_dataset, _ = load_data(args, data_config, 'valid', tokenizer, prompt_config, ratio=args.test_ratio, num=args.test_num)
        eval_func = data_config[args.data_name]["eval_func"]

        loss, acc = eval_func(args, tokenizer, data_config, eval_dataset, eval_dataloader, model, device, prompt_config, mode="test")

        log_string = "Eval result: loss: {:.6} | acc(mrr): {}".format(loss, acc)
        print_rank_0(log_string)
        save_rank_0(args, log_string)


if __name__ == "__main__":
    main()
