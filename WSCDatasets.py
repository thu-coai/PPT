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


class WSCDataset(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(WSCDataset, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):

        data = []
        enc_sizes = []
        dec_sizes = []
        self.truth = []

        with open(self.path, "r") as f:
            lines = f.readlines()

        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)

            text = d["text"]
            text = text.split(" ")
            text = " ".join(text[:d["target"]["span2_index"]] + ["*" + d["target"]["span2_text"] + "*"] + text[d["target"]["span2_index"]+1:])
            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                context = [-(i + 1) for i in range(prompt_len)] + self.tokenizer.encode(text) + [self.tokenizer.get_sentinel_id(0)]
            else:
                context = self.tokenizer.encode(text) + self.tokenizer.encode(" The pronoun " + "*" + d["target"]["span2_text"] + "*" + " refers to ") + [self.tokenizer.get_sentinel_id(0)]

            target = [0, self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode(d["target"]["span1_text"]) + [self.tokenizer.get_sentinel_id(1)]

            data.append({
                "idx": d["id"] if self.do_infer else self.idx,  
                "enc_input_ids": context,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:],
            })

            self.truth.append(d["label"])

            enc_sizes.append(len(context))
            dec_sizes.append(len(target) - 1)
            self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class WSCDatasetMan(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(WSCDatasetMan, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):

        data = []
        enc_sizes = []
        dec_sizes = []
        self.truth = []

        with open(self.path, "r") as f:
            lines = f.readlines()

        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)

            text = d["text"]
            text = text.split(" ")
            text = " ".join(text[:d["target"]["span2_index"]] + ["*" + d["target"]["span2_text"] + "*"] + text[d["target"]["span2_index"]+1:])
            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                context = [-(i + 1) for i in range(prompt_len)] + self.tokenizer.encode(text) + self.tokenizer.encode(" The pronoun " + "*" + d["target"]["span2_text"] + "*" + " refers to ") + [self.tokenizer.get_sentinel_id(0)]
            else:
                context = self.tokenizer.encode(text) + self.tokenizer.encode(" The pronoun " + "*" + d["target"]["span2_text"] + "*" + " refers to ") + [self.tokenizer.get_sentinel_id(0)]

            target = [0, self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode(d["target"]["span1_text"]) + [self.tokenizer.get_sentinel_id(1)]

            data.append({
                "idx": d["idx"],  
                "enc_input_ids": context,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:],
            })

            self.truth.append(d["label"])

            enc_sizes.append(len(context))
            dec_sizes.append(len(target) - 1)
            self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class WSCDatasetMan2(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(WSCDatasetMan2, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):

        data = []
        enc_sizes = []
        dec_sizes = []
        self.truth = []

        with open(self.path, "r") as f:
            lines = f.readlines()

        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)

            text = d["text"]
            text = text.split(" ")
            text = " ".join(text[:d["target"]["span2_index"]] + ["*" + d["target"]["span2_text"] + "*"] + text[d["target"]["span2_index"]+1:])
            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                context = [-(i + 1) for i in range(prompt_len)] + self.tokenizer.encode(text) + self.tokenizer.encode(" In the previous sentence, the pronoun " + "*" + d["target"]["span2_text"] + "*" + " refers to ") + [self.tokenizer.get_sentinel_id(0)]
            else:
                context = self.tokenizer.encode(text) + self.tokenizer.encode(" In the previous sentence, the pronoun " + "*" + d["target"]["span2_text"] + "*" + " refers to ") + [self.tokenizer.get_sentinel_id(0)]

            target = [0, self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode(d["target"]["span1_text"]) + [self.tokenizer.get_sentinel_id(1)]

            data.append({
                "idx": d["idx"],  
                "enc_input_ids": context,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:],
            })

            self.truth.append(d["label"])

            enc_sizes.append(len(context))
            dec_sizes.append(len(target) - 1)
            self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len