import torch

from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
from samplers import DistributedBatchSampler
from torch.nn import CrossEntropyLoss
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from tqdm import tqdm
import random
import numpy as np
import os
import argparse
import json
import torch.distributed as dist
import multiprocessing
import pickle

from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig


class Encoder(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, docs):
        docs = docs.strip().split("<@x(x!>")
        contexts, raw_docs = [], []
        for doc in docs:
            if len(doc) > 10 and "\t\t" not in doc:
                doc = doc[:10000]
                context = self.tokenizer.encode(doc)
                context = context[:256]
                if len(context) > 5:
                    contexts.append(context)
                    raw_docs.append(doc)
        return contexts, raw_docs


def build_data(tokenizer, save_prefix, path):
    with open(path, "r") as f:
        lines = f.readlines()
    encoder = Encoder(tokenizer)
    pool = multiprocessing.Pool(48)

    encoded_docs = pool.imap_unordered(encoder.encode, lines, 40)

    data, raw_lines = [], []

    for i, (contexts, raw_docs) in enumerate(encoded_docs):
        data.extend(contexts)
        raw_lines.extend(raw_docs)
        if i % 1000 == 0:
            print_log("processing {}".format(i))

    pool.close()

    with open(save_prefix + "_data.pkl", "wb") as f:
        pickle.dump(data, f)
    
    with open(save_prefix + "_lines.txt", "w") as f:
        for line in raw_lines:
            f.write(line + "\n")


class RawDataset(Dataset):
    def __init__(self, tokenizer: RobertaTokenizer, load_prefix):
        super().__init__()

        self.tokenizer = tokenizer

        with open(load_prefix + "_data.pkl", "rb") as f:
            self.data = pickle.load(f)

        with open(load_prefix + "_lines.txt", "r") as f:
            self.raw_lines = f.readlines()

        self.pad_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], index

    def collate(self, samples):
        bs = len(samples)
        sizes = [len(s[0]) for s in samples]
        max_size = max(sizes)
        batch = {
            "input_ids": torch.ones(bs, max_size, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, max_size, max_size),
        }

        no_model_batch = {
            "idx": torch.zeros(bs, dtype=torch.long)
        }

        for i, s in enumerate(samples):
            input_len = len(s[0])
            batch["input_ids"][i][:input_len] = torch.tensor(s[0], dtype=torch.long)
            batch["attention_mask"][i][:input_len, :input_len] = 1.0
            no_model_batch["idx"][i] = s[1]

        return batch, no_model_batch


def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def print_log(message):
    if dist.get_rank() == 0:
        print(message)


def save_log(args, message):
    if dist.get_rank() == 0:
        with open(os.path.join(args.save, "log.txt"), "a") as f:
            f.write(message + "\n")
            f.flush()


def init_distributed(args, dist_backend="nccl"):
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)
    if not torch.distributed.is_initialized():
        print("Initializing torch distributed with backend: {}".format(dist_backend))
        dist.init_process_group(backend=dist_backend)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--save", type=str)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--cache_prefix", type=str)

    args = parser.parse_args()

    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))

    set_random_seed(100)

    init_distributed(args)

    device = torch.cuda.current_device()

    config = RobertaConfig.from_pretrained(args.model_path)
    config.num_labels = 5

    model = RobertaForSequenceClassification.from_pretrained(args.model_path, config=config)
    model = model.to(device)
    i = torch.cuda.current_device()
    model = DDP(model, device_ids=[i], output_device=i)

    tokenizer = RobertaTokenizer.from_pretrained(args.model_path)

    world_size = dist.get_world_size()
    # Data parallel arguments.
    global_eval_batch_size = args.eval_batch_size * world_size
    rank = dist.get_rank()    

    if dist.get_rank() == 0:
        build_data(tokenizer, args.cache_prefix, args.data_path)

    dist.barrier()

    eval_dataset = RawDataset(tokenizer, args.cache_prefix)
    eval_sampler = DistributedBatchSampler(sampler=SequentialSampler(eval_dataset), batch_size=global_eval_batch_size, drop_last=True, rank=rank, world_size=world_size)
    eval_dataloader = DataLoader(eval_dataset, batch_sampler=eval_sampler, collate_fn=eval_dataset.collate, pin_memory=True)

    print("OK")

    all_comm = []
    model.eval()
    total_num = len(eval_dataloader)
    for i, (batch, no_model_batch) in enumerate(eval_dataloader):
        for k in batch:
            batch[k] = batch[k].to(device)
        for k in no_model_batch:
            no_model_batch[k] = no_model_batch[k].to(device)

        outputs = model(**batch)
        logits = outputs.logits

        probs = torch.softmax(logits, dim=-1)
        values, preds = torch.max(probs, dim=-1)
        idxs = no_model_batch["idx"]

        idxs = idxs.float()
        preds = preds.float()

        tensor_to_comm = torch.stack([idxs, preds, values], dim=0)        
        gathered_comm = [torch.zeros_like(tensor_to_comm) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_comm, tensor_to_comm)
        tensor_to_comm = torch.cat(gathered_comm, dim=1)

        all_comm.append(tensor_to_comm)

        if i % 100 == 0:
            print_log("{}/{}".format(i, total_num))

    all_idx, all_preds, all_values = torch.cat(all_comm, dim=1).cpu().tolist()
    all_idx = [int(x) for x in all_idx]
    all_preds = [int(x) for x in all_preds]

    if dist.get_rank() == 0:
        with open(args.save, "w") as f:
            for pred, idx, value in zip(all_preds, all_idx, all_values):
                if pred == 0:
                    if value > 0.95:
                        f.write(str(idx) + "\t\t" + eval_dataset.raw_lines[idx].strip() + "\t\t" + str(pred) + "\t\t" + str(value) + "\n")
                elif pred == 4:
                    if value > 0.7:
                        f.write(str(idx) + "\t\t" + eval_dataset.raw_lines[idx].strip() + "\t\t" + str(pred) + "\t\t" + str(value) + "\n")
                else:
                    if value > 0.5:
                        f.write(str(idx) + "\t\t" + eval_dataset.raw_lines[idx].strip() + "\t\t" + str(pred) + "\t\t" + str(value) + "\n")

if __name__ == "__main__":
    main()