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


class DBPediaDataset(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(DBPediaDataset, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):

        self.label_map = [
            349,
            1073,
            2377,
            17893,
            828,
            5127,
            740,
            1405,
            3309,
            2586,
            1475,
            2306,
            814,
            484
        ]

        data = []
        enc_sizes = []
        dec_sizes = []

        with open(self.path, "r") as f:
            lines = f.readlines()

        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)
            sid = self.tokenizer.encode(d["sentence"])

            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                context = [-(i + 1) for i in range(prompt_len)] + sid + [self.tokenizer.get_sentinel_id(0)]
            else:
                context = sid + self.tokenizer.encode("It was ") + [self.tokenizer.get_sentinel_id(0)] + [5]

            target = [0, self.tokenizer.get_sentinel_id(0)] + [self.label_map[d["label"]]]

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


class DBPediaDatasetMC(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(DBPediaDatasetMC, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):

        self.label_map = [
            349,
            1073,
            2377,
            17893,
            828,
            5127,
            740,
            1405,
            3309,
            2586,
            1475,
            2306,
            814,
            484
        ]

        self.number_map = [
            188, # A 1
            279,
            254,
            308,
            427,
            371, # F 6
            517,
            566,
            196,
            683,
            439,
            434,
            329, # M 13
            567,
            667,
            345,
            2247, # Q 17
        ]

        data = []
        enc_sizes = []
        dec_sizes = []

        options = []
        for i, option in enumerate(self.label_map):
            options.extend([self.number_map[i], 5] + self.tokenizer.encode("It was ") + [option] + [5])

        with open(self.path, "r") as f:
            lines = f.readlines()

        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)
            sid = self.tokenizer.encode(d["sentence"])

            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                context = [-(i + 1) for i in range(prompt_len)] + sid + [58] + options + [58] + self.tokenizer.encode("The correct one is ") + [self.tokenizer.get_sentinel_id(0)]
            else:
                pass

            target = [0, self.tokenizer.get_sentinel_id(0)] + [self.label_map[d["label"]]]

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

