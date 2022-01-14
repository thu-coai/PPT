from CBDatasets import CBDatasetGenTemp
from SST5Datasets import SST5DatasetGenTemp
from SST2Datasets import SST2DatasetGenTemp
from RACEDatasets import RACEDatasetGenTemp
from BoolQDatasets import BoolQDatasetGenTemp
from RTEDatasets import RTEDatasetGenTemp
import os
import random
from numpy.core.fromnumeric import mean
import torch
import json
import shutil

from torch.utils import data
import torch.nn.functional as F
import mpu
from utils import save_checkpoint
from utils import print_args
from utils import print_rank_0, save_rank_0
from utils import setup_model_and_optimizer, set_random_seed, initialize_distributed
import deepspeed

from arguments import get_args
from tokenization_t5 import EncDecTokenizer

from samplers import DistributedBatchSampler

from torch.utils.data import DataLoader, SequentialSampler

import torch.distributed as dist

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


def load_data(args, data_config, data_type, tokenizer, prompt_config=None, ratio=1, num=-1):
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
        prompt_config=prompt_config)

    sampler = SequentialSampler(dataset)
    batch_sampler = DistributedBatchSampler(sampler=sampler,
                                            batch_size=global_batch_size,
                                            drop_last=True,
                                            rank=rank,
                                            world_size=world_size)

    data_loader = DataLoader(dataset,
                             batch_sampler=batch_sampler,
                             num_workers=num_workers,
                             pin_memory=True,
                             collate_fn=dataset.collate)

    # Torch dataloader.
    return data_loader, dataset


def forward_step(args, model_batch, no_model_batch, model, device, keep_enc_hidden=False, do_infer=False):
    # if torch.distributed.get_rank() == 0:
    #     print(model_batch["enc_input_ids"][0])
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
        # if torch.distributed.get_rank() == 0:
        #     print(losses)

        loss_mask = no_model_batch["loss_mask"]
        losses = (losses * loss_mask).sum(-1) / loss_mask.sum(-1)
        loss = losses.mean()

        forw_out["loss"] = loss
        forw_out["loss_batch"] = losses
    
    return forw_out


def main():
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
            for t in ["enc", "dec"]:
                prompt_config[t]["init_ids"] = tokenizer.encode(prompt_config[t]["init_tokens"])
                pad_num = prompt_config[t]["prompt_len"] - len(prompt_config[t]["init_ids"])
                if init_from_vocab:
                    extra_id_list = [1273, 688, 4317, 18765, 16607, 2185, 14129, 11718, 6867, 16647, 21633, 16499, 11319, 14337, 3410, 23854, 21214, 24715, 17983, 8594, 18178, 10709, 9311, 14029, 2253, 10721, 7270, 14539, 19058, 6155, 18038, 20189, 26141, 5911, 5416, 24930, 17821, 5300, 11749, 12420, 2506, 8058, 18713, 14109, 3546, 2473, 14497, 4838, 15189, 20389, 22726, 22736, 1729, 3891, 3708, 14859, 19889, 20216, 14385, 20196, 14892, 9909, 15255, 15579, 25369, 15556, 21911, 389, 17297, 7668, 11825, 12037, 22548, 23342, 17876, 21091, 11246, 21048, 12660, 1239, 17118, 14809, 8821, 12683, 15571, 25274, 24743, 17935, 18640, 21428, 21936, 10781, 19864, 17113, 11648, 9248, 24316, 8105, 14614, 13714]
                    # extra_id_list = extra_id_list[:pad_num]
                    # extra_id_list = [286, 581, 919, 944, 341, 984, 59, 706, 721, 747, 713, 249, 634, 534, 515, 187, 282, 672, 616, 298, 849, 977, 666, 911, 326, 440, 568, 948, 516, 276, 723, 777, 99, 393, 638, 239, 97, 222, 77, 70, 302, 370, 391, 301, 329, 529, 417, 752, 588, 877, 955, 365, 629, 622, 668, 252, 688, 184, 741, 829, 34, 129, 664, 414, 925, 382, 648, 520, 369, 163, 930, 903, 857, 495, 388, 266, 197, 557, 693, 205, 92, 808, 824, 109, 19, 253, 727, 978, 699, 295, 801, 719, 612, 757, 619, 736, 785, 300, 816, 853]
                    # extra_id_list = [353, 596, 400, 292, 738, 161, 872, 385, 491, 784, 859, 537, 200, 961, 942, 183, 335, 547, 592, 395, 756, 549, 436, 958, 867, 654, 245, 723, 820, 391, 677, 344, 936, 155, 402, 900, 314, 546, 26, 235, 219, 584, 182, 58, 278, 929, 422, 21, 371, 462, 232, 352, 852, 286, 360, 66, 660, 40, 680, 516, 83, 591, 17, 148, 446, 518, 101, 339, 393, 211, 340, 268, 542, 108, 438, 359, 102, 599, 272, 935, 334, 868, 891, 467, 817, 450, 52, 481, 578, 120, 620, 791, 269, 678, 498, 740, 275, 68, 854, 11]
                    print(extra_id_list)
                    prompt_config[t]["init_ids"].extend(extra_id_list)
                else:
                    prompt_config[t]["init_ids"].extend(tokenizer.convert_tokens_to_ids([prompt_config[t]["default_init_token"] for _ in range(pad_num)]))
                prompt_config[t]["init_ids"] = torch.tensor(prompt_config[t]["init_ids"], dtype=torch.long).to(device)

    data_config = {
        "rte": {
            "dataset": RTEDatasetGenTemp,
            "cache_path": None,
        },
        "cb": {
            "dataset": CBDatasetGenTemp,
            "cache_path": None,
        },
        "boolq": {
            "dataset": BoolQDatasetGenTemp,
            "cache_path": None,
        },
        "race": {
            "dataset": RACEDatasetGenTemp,
            "cache_path": None,
        },
        "sst2": {
            "dataset": SST2DatasetGenTemp,
            "cache_path": None
        },
        "sst5": {
            "dataset": SST5DatasetGenTemp,
            "cache_path": None
        }
        # "c3": {
        #     "dataset": C3DatasetGenTemp,
        #     "cache_path": None
        # },
        # "poem_mc": {
        #     "dataset": PoemMCDatasetGenTemp,
        #     "cache_path": None
        # }
    }

    data_loader, dataset = load_data(args, data_config, "train", tokenizer, prompt_config=prompt_config, ratio=args.train_ratio, num=args.train_num)

    # Model, optimizer, and learning rate.
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args, tokenizer.vocab_size, ds_config, prompt_config)

    for model_batch, no_model_batch in data_loader:
        print(dist.get_rank())
        print(model_batch["enc_input_ids"])
        forw_out = forward_step(args, model_batch, no_model_batch, model, device, keep_enc_hidden=True, do_infer=True)
        max_length = 20
        enc_hidden_states = forw_out["enc_hidden_states"]

        # for generating responses
        # we only use the <go> token, so truncate other tokens
        dec_input_ids = model_batch['dec_input_ids'][..., :1]
        dec_attention_mask = model_batch['dec_attention_mask'][..., :1, :1]
        # # we use past_key_values, so only the current token mask is needed
        cross_attention_mask = model_batch['cross_attention_mask'][..., :1, :]

        bs = model_batch['enc_input_ids'].size(0)

        output_ids = model_batch['enc_input_ids'].new_zeros([model_batch['enc_input_ids'].size(0), 0])
        past_key_values = None

        total_mask_num = torch.sum((model_batch["enc_input_ids"][0] >= 32000).long(), dim=0)
        if dist.get_rank() == 0:
            print(total_mask_num)
        mask_num = 0
        for _ in range(max_length):
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
            dist.all_gather(gathered_lm_logits, lm_logits.data, mpu.get_model_parallel_group())
            lm_logits = torch.cat(gathered_lm_logits, dim=-1)

            gathered_lm_logits = [torch.zeros_like(lm_logits).to(device) for _ in range(mpu.get_data_parallel_world_size())]
            dist.all_gather(gathered_lm_logits, lm_logits.data, mpu.get_data_parallel_group())
            lm_logits = torch.cat(gathered_lm_logits, dim=0)

            next_token_logits = lm_logits[:, -1, :] / args.temperature # bs * vocab_size
            mean_next_token_logits = torch.mean(next_token_logits, dim=0)

            # mean_next_token_logits[0] = -100
            # mean_next_token_logits[3] = -100
            # mean_next_token_logits[7] = -100
            # mean_next_token_logits[10] = -100
            # mean_next_token_logits[41] = -100
            # mean_next_token_logits[443] = -100
            # mean_next_token_logits[1353] = -100
            # mean_next_token_logits[2009] = -100
            # mean_next_token_logits[13237] = -100

            next_token = torch.argmax(mean_next_token_logits, dim=-1)
            tokens_to_add = next_token.repeat(bs, 1)

            dec_input_ids = tokens_to_add
            output_ids = torch.cat([output_ids, tokens_to_add], dim=-1)
            # let the current token attend to all previous tokens
            dec_attention_mask = torch.cat([dec_attention_mask, dec_attention_mask[:, :, :, -1:]], dim=-1)

            if dist.get_rank() == 0:
                print("next_token", next_token)
                print("output_ids", output_ids[0])

            if next_token >= 32000:
                mask_num += 1

            if mask_num >= total_mask_num + 1:
                break

        break # only one iter
    
    if dist.get_rank() == 0:
        print(tokenizer.decode(output_ids[0].cpu().tolist()))


if __name__ == "__main__":
    main()