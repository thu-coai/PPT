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

from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from transformers import AdamW, get_linear_schedule_with_warmup


class SST5Dataset(Dataset):
    def __init__(self, tokenizer: RobertaTokenizer, path):
        super().__init__()
        with open(path, "r") as f:
            lines = f.readlines()

        self.data = []
        all_line_num = len(lines)
        for i, line in enumerate(lines):
            d = json.loads(line.strip())
            context = tokenizer.encode(d["sentence"])
            context = context[:511]
            label = int(d["label"])
            self.data.append({
                "context": context,
                "label": label
            })

            if i % 10000 == 0:
                print("Preprocessed {}/{}".format(i, all_line_num))

        self.pad_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate(self, samples):
        bs = len(samples)
        sizes = [len(s["context"]) for s in samples]
        max_size = max(sizes)
        batch = {
            "input_ids": torch.ones(bs, max_size, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, max_size, max_size),
            "labels": torch.zeros(bs, dtype=torch.long)
        }

        for i, s in enumerate(samples):
            input_len = len(s["context"])
            batch["input_ids"][i][:input_len] = torch.tensor(s["context"], dtype=torch.long)
            batch["attention_mask"][i][:input_len, :input_len] = 1.0
            batch["labels"][i] = s["label"]

        return batch


def evaluate(args, tokenizer, model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in tqdm(dataloader, desc="Evaluating"):
        for k in batch:
            batch[k] = batch[k].to(device)

        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits

        preds = torch.argmax(logits, dim=-1)
        labels = batch["labels"]
        all_preds.append(preds)
        all_labels.append(labels)

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    gathered_all_preds = [torch.zeros_like(all_preds) for _ in range(dist.get_world_size())]
    gathered_all_labels = [torch.zeros_like(all_labels) for _ in range(dist.get_world_size())]
    
    dist.all_gather(gathered_all_preds, all_preds)
    dist.all_gather(gathered_all_labels, all_labels)

    all_preds = torch.cat(gathered_all_preds, dim=0).cpu().tolist()
    all_labels = torch.cat(gathered_all_labels, dim=0).cpu().tolist()

    acc = acc_metric(args, all_preds, all_labels)

    return acc
    

def acc_metric(args, all_preds, all_labels):
    acc = sum([int(p == l) for p, l in zip(all_preds, all_labels)]) / len(all_preds)
    
    with open(os.path.join(args.save, "{}.txt".format(acc)), "w") as f:
        for p, l in zip(all_preds, all_labels):
            f.write(str(p) + "\t\t" + str(l) + "\n")
            f.write("\n")

    return acc


def train(args, tokenizer, model, optimizer, scheduler, train_dataloader, valid_dataloader, device):
    log_loss = 0
    step = 0
    for e in range(args.epochs):
        model.train()
        for batch in tqdm(train_dataloader):
            for k in batch:
                batch[k] = batch[k].to(device)
            
            outputs = model(**batch)
            loss = outputs.loss
            print_log("step train loss {}".format(loss))
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            log_loss += loss.item()
            if step % args.log_interval == 0:
                msg = "epoch {} step {} avg train_loss {}".format(e, step, log_loss / args.log_interval)
                print_log(msg)
                save_log(args, msg)
                log_loss = 0
            step += 1
        
        acc = evaluate(args, tokenizer, model, valid_dataloader, device)
        msg = "valid acc: {}".format(acc)
        print_log(msg)
        save_log(args, msg)
        if dist.get_rank() == 0:
            save_model(args, tokenizer, model, os.path.join(args.save, "epoch_{}".format(e)))


def save_model(args, tokenizer, model, path):
    if isinstance(model, DDP):
        model = model.module
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


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
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warm_up", type=float, default=0.01)
    parser.add_argument("--save", type=str)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--log_interval", type=int, default=10)

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
    global_batch_size = args.batch_size * world_size
    global_eval_batch_size = args.eval_batch_size * world_size
    rank = dist.get_rank()    

    train_dataset = SST5Dataset(tokenizer, os.path.join(args.data_path, "train.jsonl"))
    train_sampler = DistributedBatchSampler(sampler=RandomSampler(train_dataset), batch_size=global_batch_size, drop_last=True, rank=rank, world_size=world_size)
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=train_dataset.collate, pin_memory=True)

    valid_dataset = SST5Dataset(tokenizer, os.path.join(args.data_path, "dev.jsonl"))
    valid_sampler = DistributedBatchSampler(sampler=SequentialSampler(valid_dataset), batch_size=global_eval_batch_size, drop_last=True, rank=rank, world_size=world_size)
    valid_dataloader = DataLoader(valid_dataset, batch_sampler=valid_sampler, collate_fn=valid_dataset.collate, pin_memory=True)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_train_steps = args.epochs * len(train_dataloader)
    wm_steps = num_train_steps * args.warm_up
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=wm_steps, num_training_steps=num_train_steps)

    train(args, tokenizer, model, optimizer, scheduler, train_dataloader, valid_dataloader, device)


if __name__ == "__main__":
    main()