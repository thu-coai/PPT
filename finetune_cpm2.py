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

from QADatasets import QADataset
from YahooDatasets import YahooDataset, YahooDatasetMC
from DBPediaDatasets import DBPediaDataset, DBPediaDatasetMC
from SST5Datasets import SST5Dataset, SST5DatasetFromLM, SST5DatasetFromMC, SST5DatasetFromMCMan, SST5DatasetGenRandom, SST5DatasetGenVocab, SST5DatasetMan
from SST2MCDatasets import SST2MCDataset
from SST2Datasets import SST2Dataset, SST2DatasetFromLM, SST2DatasetFromMC, SST2DatasetFromMCMan, SST2DatasetGenRandom, SST2DatasetGenRaw, SST2DatasetGenVocab, SST2DatasetMan, SST2DatasetMan2, SST2DatasetMan3, SST2DatasetManVocab2, SST2DatasetManVocab3, SST2DatasetManVocab4
from RACEDatasets import RACEDataset, RACEDataset2, RACEDatasetFromLM, RACEDatasetFromPretrain, RACEDatasetFromPretrainLabel2, RACEDatasetFromPretrainLabel3, RACEDatasetFromPretrainLabel4, RACEDatasetMan, RACEHDatasetGenRandom, RACEHDatasetGenVocab, RACEMDatasetGenRandom, RACEMDatasetGenVocab
from WSCDatasets import WSCDataset, WSCDatasetMan, WSCDatasetMan2
from CBDatasets import CBDataset, CBDatasetFromLM, CBDatasetFromPretrainUni, CBDatasetGenRandom, CBDatasetGenVocab, CBDatasetLable2, CBDatasetLable3, CBDatasetLable4, CBDatasetMan
from COPADatasets import COPADataset
from RTEDatasets import RTEDataset, RTEDatasetFromLM, RTEDatasetFromPretrainUni, RTEDatasetGenRandom, RTEDatasetGenVocab, RTEDatasetMan
import os
import re
import random
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

from CPM2Datasets import CPM2Dataset
from BoolQDatasets import BoolQDataset, BoolQDatasetFromLM, BoolQDatasetFromPretrainUni, BoolQDatasetGenRandom, BoolQDatasetGenRaw, BoolQDatasetGenVocab, BoolQDatasetMan, BoolQDatasetMan2, BoolQDatasetMan3, BoolQDatasetManVocab2, BoolQDatasetManVocab3, BoolQDatasetManVocab4

import torch.nn.functional as F

from generation_metrics import Metric

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
          train_dataset, train_dataloader, dev_dataset, dev_dataloader, device, random_sampler: RandomSampler, prompt_config):
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
                eval_loss, acc = eval_func(args, tokenizer, data_config, dev_dataset, dev_dataloader, model, device, prompt_config, mode="dev")
                model.train()
                log_string = prefix + " eval_loss: " + str(eval_loss) + " | eval acc(mrr, f1): " + str(acc)
                print_rank_0(log_string)
                save_rank_0(args, log_string)

                if args.max_save > 0:
                    i = 0
                    while i < len(best_accs):
                        if best_accs[i][1] < acc:
                            break
                        i += 1
                    if len(best_accs) < args.max_save or i < len(best_accs):
                        best_accs.insert(i, (global_step, acc))
                        if len(best_accs) > args.max_save:
                            step_to_be_rm, acc_to_be_rm = best_accs[-1]
                            if torch.distributed.get_rank() == 0:
                                shutil.rmtree(os.path.join(args.save, "acc_{}_{:.3}".format(step_to_be_rm, acc_to_be_rm)))
                        save_checkpoint(global_step, model, optimizer, lr_scheduler, args, save_dir=os.path.join(args.save, "acc_{}_{:.3}".format(global_step, acc)))
                        best_accs = best_accs[:args.max_save]

            step += 1
            if step % args.gradient_accumulation_steps == 0:
                global_step += 1

    return global_step


def evaluate(args, tokenizer: EncDecTokenizer, data_config, eval_dataset: CPM2Dataset, eval_data_loader, model, device, prompt_config, mode='dev'):
    """Evaluation."""

    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_loss = 0.0
    step = 0

    all_idx = []
    all_preds = []
    all_labels = []
    all_true_scores = []

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


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    batch_size = logits.size()[0]
    if top_p > 0.0:
        logits=logits.view(batch_size, -1).contiguous()
        for logit in logits:
            sorted_logits, sorted_indices = torch.sort(logit, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logit[indices_to_remove] = filter_value

        logits=logits.view(batch_size, -1).contiguous()

    return logits


def evaluate_gen(args, tokenizer: EncDecTokenizer, data_config, eval_dataset: CPM2Dataset, eval_data_loader, model, device, prompt_config, mode="dev"):
    """Evaluation."""

    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_loss = 0.0
    step = 0

    all_preds = []
    all_labels = []
    all_idx = []

    with torch.no_grad():
        for model_batch, no_model_batch in eval_data_loader:
            
            forw_out = forward_step(args, model_batch, no_model_batch, model, device, keep_enc_hidden=True, do_infer=(mode=="infer"))
            loss = forw_out["loss"].item() if "loss" in forw_out else 0
            total_loss += loss

            enc_hidden_states = forw_out["enc_hidden_states"]

            dec_prompt_len = 0
            if prompt_config is not None:
                dec_prompt_len = prompt_config["dec"]["prompt_len"]
            # for generating responses
            # we only use the <go> token, so truncate other tokens
            dec_input_ids = model_batch['dec_input_ids'][..., :1 + dec_prompt_len]
            dec_attention_mask = model_batch['dec_attention_mask'][..., :1 + dec_prompt_len, :1 + dec_prompt_len]
            # # we use past_key_values, so only the current token mask is needed
            cross_attention_mask = model_batch['cross_attention_mask'][..., :1 + dec_prompt_len, :]

            unfinished_sents = model_batch['enc_input_ids'].new(model_batch['enc_input_ids'].size(0)).fill_(1)
            output_ids = model_batch['enc_input_ids'].new_zeros([model_batch['enc_input_ids'].size(0), 0])
            past_key_values = None

            gen_len = 0
            while gen_len < args.out_seq_length:
                if unfinished_sents.max() == 0:
                    tokens_to_add = tokenizer.pad_id * (1 - unfinished_sents)
                    output_ids = torch.cat([output_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

                else:
                    dec_outputs = model(
                        dec_input_ids=dec_input_ids,
                        dec_attention_mask=dec_attention_mask,
                        cross_attention_mask=cross_attention_mask,
                        enc_hidden_states=enc_hidden_states,
                        past_key_values=past_key_values,
                    )
                    lm_logits = dec_outputs["lm_logits"]
                    past_key_values = dec_outputs['past_key_values']

                    gathered_lm_logits = [torch.zeros_like(lm_logits).to(device) for _ in range(mpu.get_model_parallel_world_size())]
                    torch.distributed.all_gather(gathered_lm_logits, lm_logits.data, mpu.get_model_parallel_group())

                    lm_logits = torch.cat(gathered_lm_logits, dim=-1)
                    next_token_logits = lm_logits[:, -1, :] / args.temperature
                    if args.top_k is None and args.top_p is None:
                        next_token = torch.argmax(next_token_logits, dim=-1)
                    else:
                        next_token_logscores = top_k_logits(next_token_logits, top_k=args.top_k, top_p=args.top_p)
                        probs = F.softmax(next_token_logscores, dim=-1)
                        next_token = torch.multinomial(probs.float(), num_samples=1).squeeze(1)
                    tokens_to_add = next_token * unfinished_sents + tokenizer.pad_id * (1 - unfinished_sents)
                    dec_input_ids = tokens_to_add.unsqueeze(-1)
                    output_ids = torch.cat([output_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
                    dec_attention_mask = torch.cat([dec_attention_mask[:, :, -1:, :], dec_attention_mask[:, :, -1:, -1:]], dim=-1)
                    cross_attention_mask = cross_attention_mask[:, :, -1:, :]

                gen_len += 1
                unfinished_sents.mul_(tokens_to_add.ne(tokenizer.get_sentinel_id(1)).long())
            
            gathered_preds = [torch.zeros_like(output_ids) for _ in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather(gathered_preds, output_ids, mpu.get_data_parallel_group())
            all_preds.extend(gathered_preds)
            
            gathered_idx = [torch.zeros_like(no_model_batch["idx"]) for _ in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather(gathered_idx, no_model_batch["idx"].contiguous(), mpu.get_data_parallel_group())
            all_idx.extend(gathered_idx)

            no_model_batch["labels"] = no_model_batch["labels"][:, dec_prompt_len:].contiguous()

            gathered_labels = [torch.zeros_like(no_model_batch["labels"]) for _ in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather(gathered_labels, no_model_batch["labels"], mpu.get_data_parallel_group())
            all_labels.extend(gathered_labels)

            step += 1

    total_loss /= step

    all_idx = torch.cat(all_idx, dim=0).cpu().tolist()
    all_preds = torch.cat(all_preds, dim=0).cpu().tolist()
    all_preds = [e[:e.index(tokenizer.pad_id)] if tokenizer.pad_id in e else e for e in all_preds]
    all_labels = torch.cat(all_labels, dim=0).cpu().tolist()
    all_labels = [e[:e.index(tokenizer.pad_id)] if tokenizer.pad_id in e else e for e in all_labels]

    eval_metrc = data_config[args.data_name]["eval_metric"]
    res = eval_metrc(args, tokenizer, all_preds, all_labels, eval_dataset)

    return total_loss, res


def gen_metric(args, tokenizer: EncDecTokenizer, all_preds, all_labels):
    print("Doing gen metric")
    metric = Metric(tokenizer)
    for l, p in zip(all_labels, all_preds):
        l = list(tokenizer.decode(l[1:-1]))
        p = list(tokenizer.decode(p[1:-1]))
        metric.forword([list(map(str, l))], list(map(str, p)))
    
    metric_res, *_ = metric.close()

    with open(os.path.join(args.save, "{}.txt".format(metric_res["rouge-l"])), "w") as f:
        for p, l in zip(all_preds, all_labels):
            f.write(str(p) + "\t\t" + str(l) + "\n")
            f.write(tokenizer.decode(p) + "\t\t" + tokenizer.decode(l) + "\n\n")

    return metric_res


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


def wsc_metric(args, tokenizer: EncDecTokenizer, all_preds, all_labels, eval_dataset):
    all_task_preds = []
    all_preds = [tokenizer.decode(x[1:-1]) for x in all_preds]
    all_labels = [tokenizer.decode(x[1:-1]) for x in all_labels]
    for preds, labels in zip(all_preds, all_labels):
        preds = preds.lower().strip()
        preds = [w for w in re.split('[^a-zA-Z]', preds) if w]
        labels = labels.lower().strip()
        labels = [w for w in re.split('[^a-zA-Z]', labels) if w]

        # compare outputs
        if all(x in preds for x in labels) or all(x in labels for x in preds):
            all_task_preds.append(True)
        else:
            all_task_preds.append(False)

    all_truth = eval_dataset.truth
    acc = sum([int(p == l) for p, l in zip(all_task_preds, all_truth)]) / len(all_preds)

    with open(os.path.join(args.save, "{}.txt".format(acc)), "w") as f:
        for p, l, tp, t in zip(all_preds, all_labels, all_task_preds, all_truth):
            f.write(str(p) + "\t\t" + str(l) + "\t\t" + str(tp) + "\t\t" + str(t) + "\n")
            f.write("\n")

    return acc


class Node(object):
    def __init__(self, hidden, previous_node, decoder_input, attn, cross_attn, log_prob, length, past_key_values):
        self.hidden = hidden
        self.previous_node = previous_node
        self.decoder_input = decoder_input
        self.attn = attn
        self.cross_attn = cross_attn
        self.log_prob = log_prob
        self.past_key_values = past_key_values
        self.length = length

from queue import Queue

def evaluate_beam(args, tokenizer: EncDecTokenizer, data_config, eval_dataset: CPM2Dataset, eval_data_loader, model, device, mode="dev"):
    """Evaluation."""

    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_loss = 0.0
    step = 0

    all_preds = []
    all_labels = []
    all_idx = []

    beam_size = 4

    with torch.no_grad():
        for model_batch, no_model_batch in eval_data_loader:
            forw_out = forward_step(args, model_batch, no_model_batch, model, device, keep_enc_hidden=True, do_infer=(mode=="infer"))
            loss = forw_out["loss"].item() if "loss" in forw_out else 0
            total_loss += loss

            enc_hidden_states = forw_out["enc_hidden_states"]
            output_idxs = []
            for i in range(enc_hidden_states.shape[0]):
                root = Node(enc_hidden_states[i:i+1, ...], None, model_batch['dec_input_ids'][i:i+1, :1], model_batch['dec_attention_mask'][i:i+1, :, :1, :1], model_batch['cross_attention_mask'][i:i+1, :, :1, :], 0, 1, None)
                q = Queue()
                q.put(root)

                end_nodes = []
                while not q.empty():
                    candidates = []
                    for _ in range(q.qsize()):
                        node = q.get()

                        if node.decoder_input[0, 0].item() == tokenizer.get_sentinel_id(1) or node.length >= args.out_seq_length:
                            end_nodes.append(node)
                            continue
                            
                        dec_attention_mask = node.attn

                        dec_outputs = model(
                                dec_input_ids=node.decoder_input,
                                dec_attention_mask=dec_attention_mask,
                                cross_attention_mask=node.cross_attn,
                                enc_hidden_states=node.hidden,
                                past_key_values=node.past_key_values,
                        )

                        lm_logits = dec_outputs["lm_logits"]

                        gathered_lm_logits = [torch.zeros_like(lm_logits).to(device) for _ in range(mpu.get_model_parallel_world_size())]
                        torch.distributed.all_gather(gathered_lm_logits, lm_logits.data, mpu.get_model_parallel_group())
                        lm_logits = torch.cat(gathered_lm_logits, dim=-1)
                        next_token_logits = torch.nn.functional.log_softmax(lm_logits[:, -1, :], dim=-1)[0]

                        log_prob, indices = next_token_logits.topk(beam_size)
                        for k in range(beam_size):
                            index = indices[k].unsqueeze(0).unsqueeze(0)
                            log_p = log_prob[k].item()
                            child = Node(node.hidden, node, index, torch.cat([dec_attention_mask, dec_attention_mask[..., -1:]], dim=-1), node.cross_attn, node.log_prob + log_p, node.length + 1, dec_outputs['past_key_values'])
                            candidates.append((child.log_prob / child.length, child))
                    
                    candidates = sorted(candidates, key=lambda x:x[0], reverse=True)
                    length = min(len(candidates), beam_size)
                    for i in range(length):
                        q.put(candidates[i][1])
                max_node = sorted(end_nodes, key=lambda x: x.log_prob / x.length, reverse=True)[0]
                cur_node = max_node
                output_idx = [cur_node.decoder_input[0, 0].item()]
                while cur_node.previous_node is not None:
                    cur_node = cur_node.previous_node
                    output_idx.append(cur_node.decoder_input[0, 0].item())
                output_idx = output_idx[:-1] # sent2 .. sent1 1
                output_idx.reverse()
                output_idx += [tokenizer.pad_id] * (args.out_seq_length - len(output_idx))
                output_idxs.append(output_idx)

            output_idxs = torch.LongTensor(output_idxs).cuda()

            gathered_preds = [torch.zeros_like(output_idxs) for _ in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather(gathered_preds, output_idxs, mpu.get_data_parallel_group())
            all_preds.extend(gathered_preds)
            
            gathered_idx = [torch.zeros_like(no_model_batch["idx"]) for _ in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather(gathered_idx, no_model_batch["idx"].contiguous(), mpu.get_data_parallel_group())
            all_idx.extend(gathered_idx)

            step += 1

    total_loss /= step

    all_labels = torch.cat(all_labels, dim=0).cpu().tolist()
    all_labels = [e[:e.index(tokenizer.pad_id)] if tokenizer.pad_id in e else e for e in all_labels]
    all_preds = torch.cat(all_preds, dim=0).cpu().tolist()
    all_preds = [e[:e.index(tokenizer.pad_id)] if tokenizer.pad_id in e else e for e in all_preds]

    eval_metrc = data_config[args.data_name]["eval_metric"]
    res = eval_metrc(tokenizer, all_preds, all_labels)

    return total_loss, res


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
            init_from_vocab = prompt_config.get("init_from_vocab", False)
            init_from_label = prompt_config.get("init_from_label", False)
            if args.load_prompt is not None:
                prompt_config["load_prompt"] = args.load_prompt
            for t in ["enc", "dec"]:
                prompt_config[t]["init_ids"] = tokenizer.encode(prompt_config[t]["init_tokens"])
                pad_num = prompt_config[t]["prompt_len"] - len(prompt_config[t]["init_ids"])
                if init_from_vocab:
                    # extra_id_list = [i for i in range(0, 1000)]
                    # random.shuffle(extra_id_list)
                    # extra_id_list = extra_id_list[:pad_num]
                    # extra_id_list = [1273, 688, 4317, 18765, 16607, 2185, 14129, 11718, 6867, 16647, 21633, 16499, 11319, 14337, 3410, 23854, 21214, 24715, 17983, 8594, 18178, 10709, 9311, 14029, 2253, 10721, 7270, 14539, 19058, 6155, 18038, 20189, 26141, 5911, 5416, 24930, 17821, 5300, 11749, 12420, 2506, 8058, 18713, 14109, 3546, 2473, 14497, 4838, 15189, 20389, 22726, 22736, 1729, 3891, 3708, 14859, 19889, 20216, 14385, 20196, 14892, 9909, 15255, 15579, 25369, 15556, 21911, 389, 17297, 7668, 11825, 12037, 22548, 23342, 17876, 21091, 11246, 21048, 12660, 1239, 17118, 14809, 8821, 12683, 15571, 25274, 24743, 17935, 18640, 21428, 21936, 10781, 19864, 17113, 11648, 9248, 24316, 8105, 14614, 13714]
                    # extra_id_list = [286, 581, 919, 944, 341, 984, 59, 706, 721, 747, 713, 249, 634, 534, 515, 187, 282, 672, 616, 298, 849, 977, 666, 911, 326, 440, 568, 948, 516, 276, 723, 777, 99, 393, 638, 239, 97, 222, 77, 70, 302, 370, 391, 301, 329, 529, 417, 752, 588, 877, 955, 365, 629, 622, 668, 252, 688, 184, 741, 829, 34, 129, 664, 414, 925, 382, 648, 520, 369, 163, 930, 903, 857, 495, 388, 266, 197, 557, 693, 205, 92, 808, 824, 109, 19, 253, 727, 978, 699, 295, 801, 719, 612, 757, 619, 736, 785, 300, 816, 853] # ocnli
                    # extra_id_list = [353, 596, 400, 292, 738, 161, 872, 385, 491, 784, 859, 537, 200, 961, 942, 183, 335, 547, 592, 395, 756, 549, 436, 958, 867, 654, 245, 723, 820, 391, 677, 344, 936, 155, 402, 900, 314, 546, 26, 235, 219, 584, 182, 58, 278, 929, 422, 21, 371, 462, 232, 352, 852, 286, 360, 66, 660, 40, 680, 516, 83, 591, 17, 148, 446, 518, 101, 339, 393, 211, 340, 268, 542, 108, 438, 359, 102, 599, 272, 935, 334, 868, 891, 467, 817, 450, 52, 481, 578, 120, 620, 791, 269, 678, 498, 740, 275, 68, 854, 11] # most
                    if "sst2" in args.data_name:
                        extra_id_list = [1974, 59, 120, 814, 103, 95, 80, 114, 20, 53, 107, 145, 12, 485, 56, 38, 57, 36, 12373, 97, 99, 112, 396, 18, 73, 23, 165, 125, 113, 6, 41, 49, 5, 66, 46, 28, 385, 2850, 34, 237, 32, 68, 60, 52, 9, 7, 179, 17, 81, 128, 25, 2, 19, 210, 48, 13, 1636, 139, 150, 405, 21, 116, 143, 733, 91, 30, 31, 61, 131, 11, 43, 77, 168, 40, 207, 3, 75, 88, 16, 29, 24, 63, 42, 35, 65, 167, 15, 8, 45, 132, 44, 54, 33, 26, 233, 72, 231, 51, 78, 89] # sst2
                    elif "boolq" in args.data_name:
                        extra_id_list = [8, 34, 66, 12, 61, 189, 38, 261, 33, 84, 43, 77, 1150, 92, 138, 3, 137, 24, 1323, 117, 70, 36, 166, 15, 13, 79, 113, 52, 59, 26, 51, 29, 45, 201, 939, 16, 227, 119, 797, 32, 46, 65, 139, 2, 47, 132, 57, 1636, 68, 18, 130, 49, 60, 37, 30, 160, 134, 106, 23, 28, 88, 774, 40, 31, 163, 42, 6, 192, 224, 11, 75, 71, 86, 164, 9, 17, 165, 102, 145, 814, 337, 63, 7, 21, 94, 120, 80, 44, 112, 5, 118, 405, 41, 54, 10, 53, 89, 97, 19, 907] # boolq
                    else:
                        raise NotImplementedError
                        
                    if torch.distributed.get_rank() == 0:
                        print(extra_id_list)
                        print(tokenizer.decode(extra_id_list))
                    prompt_config[t]["init_ids"].extend(extra_id_list)
                elif init_from_label:
                    all_label_ids = prompt_config["all_label_ids"]
                    repeat_num = 100 // len(all_label_ids) + 1
                    extra_id_list = (all_label_ids * repeat_num)[:100]
                    if torch.distributed.get_rank() == 0:
                        print(extra_id_list)
                        print(tokenizer.decode(extra_id_list))
                    prompt_config[t]["init_ids"].extend(extra_id_list)
                else:
                    prompt_config[t]["init_ids"].extend(tokenizer.convert_tokens_to_ids([prompt_config[t]["default_init_token"] for _ in range(pad_num)]))
                prompt_config[t]["init_ids"] = torch.tensor(prompt_config[t]["init_ids"], dtype=torch.long).to(device)

    data_config = {
        "boolq": {
            "dataset": BoolQDataset,
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
        "copa": {
            "dataset": COPADataset,
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
        "race": {
            "dataset": RACEDataset,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "race2": {
            "dataset": RACEDataset2,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "race_man": {
            "dataset": RACEDatasetMan,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "race_from_pretrain": {
            "dataset": RACEDatasetFromPretrain,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "race_from_lm": {
            "dataset": RACEDatasetFromLM,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "race_m_gen_random": {
            "dataset": RACEMDatasetGenRandom,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "race_m_gen_vocab": {
            "dataset": RACEMDatasetGenVocab,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "race_from_pretrain_label2": {
            "dataset": RACEDatasetFromPretrainLabel2,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "race_from_pretrain_label3": {
            "dataset": RACEDatasetFromPretrainLabel3,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "race_from_pretrain_label4": {
            "dataset": RACEDatasetFromPretrainLabel4,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "race_h_gen_random": {
            "dataset": RACEHDatasetGenRandom,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "race_h_gen_vocab": {
            "dataset": RACEHDatasetGenVocab,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "boolq_man": {
            "dataset": BoolQDatasetMan,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "boolq_man_2": {
            "dataset": BoolQDatasetMan2,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "boolq_man_3": {
            "dataset": BoolQDatasetMan3,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "boolq_gen_random": {
            "dataset": BoolQDatasetGenRandom,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "boolq_man_vocab2": {
            "dataset": BoolQDatasetManVocab2,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "boolq_man_vocab3": {
            "dataset": BoolQDatasetManVocab3,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "boolq_man_vocab4": {
            "dataset": BoolQDatasetManVocab4,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "boolq_gen_raw": {
            "dataset": BoolQDatasetGenRaw,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },        
        "boolq_gen_vocab": {
            "dataset": BoolQDatasetGenVocab,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "boolq_from_pretrain_uni": {
            "dataset": BoolQDatasetFromPretrainUni,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "boolq_from_lm": {
            "dataset": BoolQDatasetFromLM,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "rte_man": {
            "dataset": RTEDatasetMan,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "rte_gen_random": {
            "dataset": RTEDatasetGenRandom,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "rte_gen_vocab": {
            "dataset": RTEDatasetGenVocab,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "rte_from_pretrain_uni": {
            "dataset": RTEDatasetFromPretrainUni,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "rte_from_lm": {
            "dataset": RTEDatasetFromLM,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "cb_man": {
            "dataset": CBDatasetMan,
            "eval_func": evaluate,
            "eval_metric": acc_f1_metric,
            "cache_path": None,
        },
        "cb_gen_random": {
            "dataset": CBDatasetGenRandom,
            "eval_func": evaluate,
            "eval_metric": acc_f1_metric,
            "cache_path": None,
        },
        "cb_gen_vocab": {
            "dataset": CBDatasetGenVocab,
            "eval_func": evaluate,
            "eval_metric": acc_f1_metric,
            "cache_path": None,
        },
        "cb_from_pretrain_uni": {
            "dataset": CBDatasetFromPretrainUni,
            "eval_func": evaluate,
            "eval_metric": acc_f1_metric,
            "cache_path": None,
        },
        "cb_from_lm": {
            "dataset": CBDatasetFromLM,
            "eval_func": evaluate,
            "eval_metric": acc_f1_metric,
            "cache_path": None,
        },
        "cb_from_pretrain_label2": {
            "dataset": CBDatasetLable2,
            "eval_func": evaluate,
            "eval_metric": acc_f1_metric,
            "cache_path": None,
        },
        "cb_from_pretrain_label3": {
            "dataset": CBDatasetLable3,
            "eval_func": evaluate,
            "eval_metric": acc_f1_metric,
            "cache_path": None,
        },
        "cb_from_pretrain_label4": {
            "dataset": CBDatasetLable4,
            "eval_func": evaluate,
            "eval_metric": acc_f1_metric,
            "cache_path": None,
        },
        "wsc": {
            "dataset": WSCDataset,
            "eval_func": evaluate_gen,
            "eval_metric": wsc_metric,
            "cache_path": None
        },
        "wsc_man": {
            "dataset": WSCDatasetMan,
            "eval_func": evaluate_gen,
            "eval_metric": wsc_metric,
            "cache_path": None
        },
        "wsc_man_2": {
            "dataset": WSCDatasetMan2,
            "eval_func": evaluate_gen,
            "eval_metric": wsc_metric,
            "cache_path": None
        },
        "sst2": {
            "dataset": SST2Dataset,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "sst2_man": {
            "dataset": SST2DatasetMan,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "sst2_man_2": {
            "dataset": SST2DatasetMan2,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "sst2_man_3": {
            "dataset": SST2DatasetMan3,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "sst2_man_vocab2": {
            "dataset": SST2DatasetManVocab2,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "sst2_man_vocab3": {
            "dataset": SST2DatasetManVocab3,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "sst2_man_vocab4": {
            "dataset": SST2DatasetManVocab4,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "sst2_from_mc": {
            "dataset": SST2DatasetFromMC,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None
        },
        "sst2_mc": {
            "dataset": SST2MCDataset,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None
        },
        "sst2_from_mc_man": {
            "dataset": SST2DatasetFromMCMan,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None
        },
        "sst2_gen_raw": {
            "dataset": SST2DatasetGenRaw,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None
        },
        "sst2_gen_random": {
            "dataset": SST2DatasetGenRandom,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None
        },
        "sst2_from_lm": {
            "dataset": SST2DatasetFromLM,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None
        },
        "sst2_gen_vocab": {
            "dataset": SST2DatasetGenVocab,
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
        "sst5_man": {
            "dataset": SST5DatasetMan,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "sst5_from_mc": {
            "dataset": SST5DatasetFromMC,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "sst5_from_mc_man": {
            "dataset": SST5DatasetFromMCMan,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "sst5_gen_random": {
            "dataset": SST5DatasetGenRandom,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None
        },
        "sst5_gen_vocab": {
            "dataset": SST5DatasetGenVocab,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None
        },
        "sst5_from_mc_man": {
            "dataset": SST5DatasetFromLM,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "sst5_from_lm": {
            "dataset": SST5DatasetFromLM,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "dbpedia": {
            "dataset": DBPediaDataset,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "dbpedia_mc": {
            "dataset": DBPediaDatasetMC,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "yahoo": {
            "dataset": YahooDataset,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "yahoo_mc": {
            "dataset": YahooDatasetMC,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "dbpedia_from_pretrain_uni": {
            "dataset": DBPediaDatasetMC,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        },
        "qa": {
            "dataset": QADataset,
            "eval_func": evaluate,
            "eval_metric": acc_metric,
            "cache_path": None,
        }
    }

    if args.do_train:
        train_dataloader, train_dataset, random_sampler = load_data(args, data_config, 'train', tokenizer, prompt_config, ratio=args.train_ratio, num=args.train_num)
        dev_dataloader, dev_dataset, _  = load_data(args, data_config, 'valid', tokenizer, prompt_config, ratio=args.dev_ratio, num=args.dev_num)
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
        train(args, data_config, tokenizer, model, optimizer, lr_scheduler, train_dataset, train_dataloader, dev_dataset, dev_dataloader, device, random_sampler, prompt_config)

    if args.do_eval:
        eval_dataloader, eval_dataset, _ = load_data(args, data_config, 'valid', tokenizer, prompt_config, ratio=args.test_ratio, num=args.test_num)
        eval_func = data_config[args.data_name]["eval_func"]

        loss, acc = eval_func(args, tokenizer, data_config, eval_dataset, eval_dataloader, model, device, prompt_config, mode="test")

        log_string = "Eval result: loss: {:.6} | acc(mrr): {}".format(loss, acc)
        print_rank_0(log_string)
        save_rank_0(args, log_string)


if __name__ == "__main__":
    main()
