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


class CBDataset(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(CBDataset, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)

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
                "contradiction": "no",
                "neutral": "maybe",
            }

            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                context = [-(i + 1) for i in range(prompt_len)] + self.tokenizer.encode(d["hypothesis"]) + [self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode(d["premise"])
            else:
                context =  self.tokenizer.encode(d["hypothesis"] + "?") + [self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode(".") + self.tokenizer.encode(d["premise"])

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


class CBDatasetMan(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(CBDatasetMan, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)

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
                "contradiction": "no",
                "neutral": "maybe",
            }

            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                context = [-(i + 1) for i in range(prompt_len)] + self.tokenizer.encode(d["hypothesis"] + "?") + [self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode(".") + self.tokenizer.encode(d["premise"])
            else:
                context =  self.tokenizer.encode(d["hypothesis"] + "?") + [self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode(".") + self.tokenizer.encode(d["premise"])

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


class CBDatasetGenTemp(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(CBDatasetGenTemp, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)

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
                "contradiction": "no",
                "neutral": "maybe",
            }

            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                context = [-(i + 1) for i in range(prompt_len)] + [self.tokenizer.get_sentinel_id(0)] + \
                                self.tokenizer.encode(d["hypothesis"]) + [self.tokenizer.get_sentinel_id(1)] + \
                                    self.tokenizer.encode(label_map[d["label"]]) + [self.tokenizer.get_sentinel_id(2)] + \
                                        self.tokenizer.encode(d["premise"]) + [self.tokenizer.get_sentinel_id(3)]
            else:
                context =  self.tokenizer.encode(d["hypothesis"]) + [self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode(".") + self.tokenizer.encode(d["premise"])

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


class CBDatasetGenRandom(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(CBDatasetGenRandom, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)

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
                "contradiction": "no",
                "neutral": "maybe",
            }

            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                context = [-(i + 1) for i in range(prompt_len)] + [5] + self.tokenizer.encode(d["hypothesis"]) + [5] + [self.tokenizer.get_sentinel_id(0)] + [5] + self.tokenizer.encode(d["premise"]) + [5]
            else:
                context =  self.tokenizer.encode(d["hypothesis"]) + [self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode(".") + self.tokenizer.encode(d["premise"])

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


class CBDatasetGenVocab(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(CBDatasetGenVocab, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)

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
                "contradiction": "no",
                "neutral": "maybe",
            }

            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                context = [-(i + 1) for i in range(prompt_len)] + [15, 7, 159, 5] + self.tokenizer.encode(d["hypothesis"]) + [5] + [self.tokenizer.get_sentinel_id(0)] + [5] + self.tokenizer.encode(d["premise"]) + [3, 2]
            else:
                context =  self.tokenizer.encode(d["hypothesis"] + "?") + [self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode(".") + self.tokenizer.encode(d["premise"])

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


class CBDatasetFromPretrainUni(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(CBDatasetFromPretrainUni, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):

        data = []
        enc_sizes = []
        dec_sizes = []

        with open(self.path, "r") as f:
            lines = f.readlines()

        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)

            label_map = {
                "contradiction": 188,
                "neutral": 279,
                "entailment": 254,
            }

            if self.prompt_config:
                choice_ids = [188, 5] + self.tokenizer.encode("no") + [5] + [279, 5] + self.tokenizer.encode("maybe") + [5] + [254, 5] + self.tokenizer.encode("yes") + [5]
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                context = [-(i + 1) for i in range(prompt_len)] + self.tokenizer.encode(d["hypothesis"]) + [58] + self.tokenizer.encode(d["premise"]) + [5] + choice_ids + [58] + self.tokenizer.encode("The correct one is ") + [self.tokenizer.get_sentinel_id(0)]
            else:
                pass

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


class CBDatasetFromLM(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(CBDatasetFromLM, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)

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
                "contradiction": "no",
                "neutral": "maybe",
            }

            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                context = [-(i + 1) for i in range(prompt_len)] + self.tokenizer.encode("hypothesis: ") + self.tokenizer.encode(d["hypothesis"]) + self.tokenizer.encode("premise: ") + self.tokenizer.encode(d["premise"])
            else:
                pass

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


class CBDatasetLable2(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(CBDatasetLable2, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):

        data = []
        enc_sizes = []
        dec_sizes = []

        with open(self.path, "r") as f:
            lines = f.readlines()

        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)

            label_map = {
                "entailment": "true",
                "contradiction": "false",
                "neutral": "neutral",
            }

            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                context = [-(i + 1) for i in range(prompt_len)] + self.tokenizer.encode(d["hypothesis"]) + [self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode(d["premise"])
            else:
                pass

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


class CBDatasetLable3(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(CBDatasetLable3, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):

        data = []
        enc_sizes = []
        dec_sizes = []

        with open(self.path, "r") as f:
            lines = f.readlines()

        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)

            label_map = {
                "entailment": "no",
                "contradiction": "yes",
                "neutral": "maybe",
            }

            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                context = [-(i + 1) for i in range(prompt_len)] + self.tokenizer.encode(d["hypothesis"]) + [self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode(d["premise"])
            else:
                pass

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


class CBDatasetLable4(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(CBDatasetLable4, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):

        data = []
        enc_sizes = []
        dec_sizes = []

        with open(self.path, "r") as f:
            lines = f.readlines()

        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)

            label_map = {
                "entailment": "dog",
                "contradiction": "cat",
                "neutral": "human",
            }

            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                context = [-(i + 1) for i in range(prompt_len)] + self.tokenizer.encode(d["hypothesis"]) + [self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode(d["premise"])
            else:
                pass

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