import torch
import json
import re
import os
import random
from torch.utils.data import Dataset
from tokenization_t5 import EncDecTokenizer
import pickle
import mpu
import math
from utils import print_rank_0, save_rank_0
from CPM2Datasets import CPM2Dataset


class RTEDataset(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(RTEDataset, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):

        data = []
        enc_sizes = []
        dec_sizes = []

        with open(self.path, "r") as f:
            lines = f.readlines()

        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)

            label_map = {
                "entailment": "yes",
                "not_entailment": "no"
            }

            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                context = [-(i + 1) for i in range(prompt_len)] + self.tokenizer.encode(d["premise"]) + [self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode(d["hypothesis"])
            else:
                context =  self.tokenizer.encode(d["hypothesis"]) + [58] + [self.tokenizer.get_sentinel_id(0)] + [5] + self.tokenizer.encode(d["premise"])

            target = [0, self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode(label_map[d["label"]])

            data.append({
                "idx": d["id"] if self.do_infer else self.idx,  
                "enc_input_ids": context,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:]
            })

            enc_sizes.append(len(context))
            dec_sizes.append(len(target) - 1)
            self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class RTEDatasetMan(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(RTEDatasetMan, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):

        data = []
        enc_sizes = []
        dec_sizes = []

        with open(self.path, "r") as f:
            lines = f.readlines()

        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)

            label_map = {
                "entailment": "yes",
                "not_entailment": "no"
            }

            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                context = [-(i + 1) for i in range(prompt_len)] + self.tokenizer.encode(d["hypothesis"]) + [58] + [self.tokenizer.get_sentinel_id(0)] + [5] + self.tokenizer.encode(d["premise"])
            else:
                context =  self.tokenizer.encode(d["hypothesis"]) + [58] + [self.tokenizer.get_sentinel_id(0)] + [5] + self.tokenizer.encode(d["premise"])

            target = [0, self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode(label_map[d["label"]])

            data.append({
                "idx": d["id"] if self.do_infer else self.idx,  
                "enc_input_ids": context,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:]
            })

            enc_sizes.append(len(context))
            dec_sizes.append(len(target) - 1)
            self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class RTEDatasetGenRandom(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(RTEDatasetGenRandom, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):

        data = []
        enc_sizes = []
        dec_sizes = []

        with open(self.path, "r") as f:
            lines = f.readlines()

        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)

            label_map = {
                "entailment": "yes",
                "not_entailment": "no"
            }

            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                context = [-(i + 1) for i in range(prompt_len)] + [5] + self.tokenizer.encode(d["hypothesis"]) + [37] + [self.tokenizer.get_sentinel_id(0)] + [18, 29, 32, 5] + self.tokenizer.encode(d["premise"]) + [37, 150, 18, 29, 32, 5]
            else:
                context =  self.tokenizer.encode(d["hypothesis"]) + [5] + [self.tokenizer.get_sentinel_id(0)] + [37] + self.tokenizer.encode(d["premise"])

            target = [0, self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode(label_map[d["label"]])

            data.append({
                "idx": d["id"] if self.do_infer else self.idx,  
                "enc_input_ids": context,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:]
            })

            enc_sizes.append(len(context))
            dec_sizes.append(len(target) - 1)
            self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class RTEDatasetGenVocab(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(RTEDatasetGenVocab, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):

        data = []
        enc_sizes = []
        dec_sizes = []

        with open(self.path, "r") as f:
            lines = f.readlines()

        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)

            label_map = {
                "entailment": "yes",
                "not_entailment": "no"
            }

            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                context = [-(i + 1) for i in range(prompt_len)] + [15, 7, 159, 5] + self.tokenizer.encode(d["hypothesis"]) + [37] + [self.tokenizer.get_sentinel_id(0)] + [5] + self.tokenizer.encode(d["premise"]) + [37, 150, 5]
            else:
                context =  self.tokenizer.encode(d["hypothesis"]) + [5] + [self.tokenizer.get_sentinel_id(0)] + [37] + self.tokenizer.encode(d["premise"])

            target = [0, self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode(label_map[d["label"]])

            data.append({
                "idx": d["id"] if self.do_infer else self.idx,  
                "enc_input_ids": context,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:]
            })

            enc_sizes.append(len(context))
            dec_sizes.append(len(target) - 1)
            self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class RTEDatasetGenTemp(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(RTEDatasetGenTemp, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):

        label_map = {
            "entailment": "yes",
            "not_entailment": "no"
        }
    
        data = []
        enc_sizes = []
        dec_sizes = []
        
        with open(self.path, "r") as f:
            lines = f.readlines()
        
        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)
            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                context = [-(i + 1) for i in range(prompt_len)] + [self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode(d["hypothesis"]) + \
                            [self.tokenizer.get_sentinel_id(1)] + self.tokenizer.encode(label_map[d["label"]]) + \
                            [self.tokenizer.get_sentinel_id(2)] + self.tokenizer.encode(d["premise"]) + [self.tokenizer.get_sentinel_id(3)]
            else:
                context = [self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode(d["hypothesis"]) + \
                            [self.tokenizer.get_sentinel_id(1)] + self.tokenizer.encode(label_map[d["label"]]) + \
                            [self.tokenizer.get_sentinel_id(2)] + self.tokenizer.encode(d["premise"]) + [self.tokenizer.get_sentinel_id(3)]

            target = [0, self.tokenizer.get_sentinel_id(0)] + (self.tokenizer.encode(label_map[d["label"]]) if not self.do_infer else [self.tokenizer.pad_id])
            if self.add_target_post:
                target += [self.tokenizer.get_sentinel_id(1)]

            data.append({
                "idx": d["id"] if self.do_infer else self.idx,  
                "enc_input_ids": context,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:]
            })

            enc_sizes.append(len(context))
            dec_sizes.append(len(target) - 1)
            self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class RTEDatasetFromPretrainUni(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(RTEDatasetFromPretrainUni, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):

        data = []
        enc_sizes = []
        dec_sizes = []

        with open(self.path, "r") as f:
            lines = f.readlines()

        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)

            label_map = {
                "not_entailment": 188,
                "entailment": 279,
            }

            if self.prompt_config:
                choice_ids = [188, 5] + self.tokenizer.encode("no") + [5] + [279, 5] + self.tokenizer.encode("yes") + [5]
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                context = [-(i + 1) for i in range(prompt_len)] + self.tokenizer.encode(d["hypothesis"]) + [58] + self.tokenizer.encode(d["premise"]) + [5] + choice_ids + [58] + self.tokenizer.encode("The correct one is ") + [self.tokenizer.get_sentinel_id(0)]
            else:
                context =  self.tokenizer.encode(d["hypothesis"]) + [58] + [self.tokenizer.get_sentinel_id(0)] + [5] + self.tokenizer.encode(d["premise"])

            target = [0, self.tokenizer.get_sentinel_id(0)] + [label_map[d["label"]]]

            data.append({
                "idx": d["id"] if self.do_infer else self.idx,  
                "enc_input_ids": context,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:]
            })

            enc_sizes.append(len(context))
            dec_sizes.append(len(target) - 1)
            self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class RTEDatasetFromLM(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(RTEDatasetFromLM, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):

        data = []
        enc_sizes = []
        dec_sizes = []

        with open(self.path, "r") as f:
            lines = f.readlines()

        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)

            label_map = {
                "entailment": "yes",
                "not_entailment": "no"
            }

            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                context = [-(i + 1) for i in range(prompt_len)] + self.tokenizer.encode("premise: ") + self.tokenizer.encode(d["premise"]) + self.tokenizer.encode("hypothesis: ") + self.tokenizer.encode(d["hypothesis"])
            else:
                context =  self.tokenizer.encode(d["hypothesis"]) + [58] + [self.tokenizer.get_sentinel_id(0)] + [5] + self.tokenizer.encode(d["premise"])

            target = [0] + self.tokenizer.encode(label_map[d["label"]])

            data.append({
                "idx": d["id"] if self.do_infer else self.idx,  
                "enc_input_ids": context,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:]
            })

            enc_sizes.append(len(context))
            dec_sizes.append(len(target) - 1)
            self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len