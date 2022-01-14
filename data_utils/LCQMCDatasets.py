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

class LCQMCDataset(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(LCQMCDataset, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):

        self.label_word_map = ["矛盾", "相似"]
    
        data = []
        enc_sizes = []
        dec_sizes = []
        
        with open(self.path, "r") as f:
            lines = f.readlines()
        
        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)
            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                first_ratio = self.prompt_config["enc"].get("ratio_list", [0.5, 0.5, 0])[0]
                med_ratio = self.prompt_config["enc"].get("ratio_list", [0.5, 0.5, 0])[1]
                if med_ratio > 0.0:
                    med_token = []
                else:
                    med_token = [18]
                context = [-(i + 1) for i in range(int(first_ratio * prompt_len))] + self.tokenizer.encode(d["sentence1"]) + med_token + [-(i + 1) for i in range(int(first_ratio * prompt_len), int((first_ratio + med_ratio) * prompt_len))] + self.tokenizer.encode(d["sentence2"]) + [-(i + 1) for i in range(int((first_ratio + med_ratio) * prompt_len), prompt_len)]
            else:
                context = [39] + self.tokenizer.encode(d["sentence1"]) + [41, 62, 39] + self.tokenizer.encode(d["sentence2"]) + [41, 11, 1348, self.tokenizer.get_sentinel_id(0)]
            target = [1, self.tokenizer.get_sentinel_id(0)] + (self.tokenizer.encode(self.label_word_map[d["label"]]) if not self.do_infer else [self.tokenizer.pad_id])
            if self.add_target_post:
                target += [self.tokenizer.get_sentinel_id(1)]

            data.append({
                "idx": d["id"] if self.do_infer else self.idx,  
                "enc_input_ids": context,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:]
            })

            if self.prompt_config:
                if self.prompt_config.get("mask_prompt2token", False):
                    p_mask = (torch.tensor(context) < 0).unsqueeze(1).repeat(1, len(context))
                    enc_attention_mask = (p_mask & p_mask.T | (~p_mask)).float()
                    data[-1].update({
                        "enc_attention_mask": enc_attention_mask 
                    })
                if self.prompt_config.get("mask_token2prompt", False):
                    p_mask = (torch.tensor(context) < 0).unsqueeze(1).repeat(1, len(context))
                    enc_attention_mask = (p_mask & p_mask.T | (~p_mask)).T.float()
                    # cross_attention_mask = (torch.tensor(context) >= 0).unsqueeze(1).repeat(1, len(target) - 1).T
                    data[-1].update({
                        "enc_attention_mask": enc_attention_mask,
                        # "cross_attention_mask": cross_attention_mask
                    })

            enc_sizes.append(len(context))
            dec_sizes.append(len(target) - 1)
            self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class LCQMCDatasetRaw(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(LCQMCDatasetRaw, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):

        self.label_word_map = ["矛盾", "相似"]
    
        data = []
        enc_sizes = []
        dec_sizes = []
        
        with open(self.path, "r") as f:
            lines = f.readlines()
        
        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)
            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                context = [-(i + 1) for i in range(int(prompt_len))] + self.tokenizer.encode(d["sentence1"]) + [self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode(d["sentence2"])
            else:
                context = [39] + self.tokenizer.encode(d["sentence1"]) + [41, 62, 39] + self.tokenizer.encode(d["sentence2"]) + [41, 11, 1348, self.tokenizer.get_sentinel_id(0)]
            target = [1, self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode(self.label_word_map[d["label"]])
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


class LCQMCDatasetTemplate(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(LCQMCDatasetTemplate, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):

        self.label_word_map = ["矛盾", "相似"]
    
        data = []
        enc_sizes = []
        dec_sizes = []
        
        with open(self.path, "r") as f:
            lines = f.readlines()
        
        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)
            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                first_ratio = self.prompt_config["enc"].get("ratio_list", [0.5, 0.5, 0])[0]
                med_ratio = self.prompt_config["enc"].get("ratio_list", [0.5, 0.5, 0])[1]
                context = [-(i + 1) for i in range(int(first_ratio * prompt_len))] + self.tokenizer.encode(d["sentence1"]) + [self.tokenizer.get_sentinel_id(0)] + [-(i + 1) for i in range(int(first_ratio * prompt_len), int((first_ratio + med_ratio) * prompt_len))] + self.tokenizer.encode(d["sentence2"]) + [-(i + 1) for i in range(int((first_ratio + med_ratio) * prompt_len), prompt_len)]
            else:
                context = [39] + self.tokenizer.encode(d["sentence1"]) + [41, 62, 39] + self.tokenizer.encode(d["sentence2"]) + [41, 11, 1348, self.tokenizer.get_sentinel_id(0)]
            target = [1, self.tokenizer.get_sentinel_id(0)] + (self.tokenizer.encode(self.label_word_map[d["label"]]) if not self.do_infer else [self.tokenizer.pad_id])
            if self.add_target_post:
                target += [self.tokenizer.get_sentinel_id(1)]

            data.append({
                "idx": d["id"] if self.do_infer else self.idx,  
                "enc_input_ids": context,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:]
            })

            if self.prompt_config:
                if self.prompt_config.get("mask_prompt2token", False):
                    p_mask = (torch.tensor(context) < 0).unsqueeze(1).repeat(1, len(context))
                    enc_attention_mask = (p_mask & p_mask.T | (~p_mask)).float()
                    data[-1].update({
                        "enc_attention_mask": enc_attention_mask 
                    })
                if self.prompt_config.get("mask_token2prompt", False):
                    p_mask = (torch.tensor(context) < 0).unsqueeze(1).repeat(1, len(context))
                    enc_attention_mask = (p_mask & p_mask.T | (~p_mask)).T.float()
                    # cross_attention_mask = (torch.tensor(context) >= 0).unsqueeze(1).repeat(1, len(target) - 1).T
                    data[-1].update({
                        "enc_attention_mask": enc_attention_mask,
                        # "cross_attention_mask": cross_attention_mask
                    })

            enc_sizes.append(len(context))
            dec_sizes.append(len(target) - 1)
            self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class LCQMCDatasetTemplateGen(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(LCQMCDatasetTemplateGen, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):

        self.label_word_map = ["矛盾", "相似"]
    
        data = []
        enc_sizes = []
        dec_sizes = []
        
        with open(self.path, "r") as f:
            lines = f.readlines()
        
        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)
            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                first_ratio = self.prompt_config["enc"].get("ratio_list", [0.5, 0.5, 0])[0]
                med_ratio = self.prompt_config["enc"].get("ratio_list", [0.5, 0.5, 0])[1]
                context = [-(i + 1) for i in range(int(first_ratio * prompt_len))] + [11] + self.tokenizer.encode(d["sentence1"]) + [16] + [self.tokenizer.get_sentinel_id(0)] + [11] + [-(i + 1) for i in range(int(first_ratio * prompt_len), int((first_ratio + med_ratio) * prompt_len))] + self.tokenizer.encode(d["sentence2"]) + [16] + [-(i + 1) for i in range(int((first_ratio + med_ratio) * prompt_len), prompt_len)]
            else:
                context = [39] + self.tokenizer.encode(d["sentence1"]) + [41, 62, 39] + self.tokenizer.encode(d["sentence2"]) + [41, 11, 1348, self.tokenizer.get_sentinel_id(0)]
            target = [1, self.tokenizer.get_sentinel_id(0)] + (self.tokenizer.encode(self.label_word_map[d["label"]]) if not self.do_infer else [self.tokenizer.pad_id])
            if self.add_target_post:
                target += [self.tokenizer.get_sentinel_id(1)]

            data.append({
                "idx": d["id"] if self.do_infer else self.idx,  
                "enc_input_ids": context,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:]
            })

            if self.prompt_config:
                if self.prompt_config.get("mask_prompt2token", False):
                    p_mask = (torch.tensor(context) < 0).unsqueeze(1).repeat(1, len(context))
                    enc_attention_mask = (p_mask & p_mask.T | (~p_mask)).float()
                    data[-1].update({
                        "enc_attention_mask": enc_attention_mask 
                    })
                if self.prompt_config.get("mask_token2prompt", False):
                    p_mask = (torch.tensor(context) < 0).unsqueeze(1).repeat(1, len(context))
                    enc_attention_mask = (p_mask & p_mask.T | (~p_mask)).T.float()
                    # cross_attention_mask = (torch.tensor(context) >= 0).unsqueeze(1).repeat(1, len(target) - 1).T
                    data[-1].update({
                        "enc_attention_mask": enc_attention_mask,
                        # "cross_attention_mask": cross_attention_mask
                    })

            enc_sizes.append(len(context))
            dec_sizes.append(len(target) - 1)
            self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class LCQMCDatasetGenTemp(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(LCQMCDatasetGenTemp, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):

        self.label_word_map = ["矛盾", "相似"]
    
        data = []
        enc_sizes = []
        dec_sizes = []
        
        with open(self.path, "r") as f:
            lines = f.readlines()
        
        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)
            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                first_ratio = self.prompt_config["enc"].get("ratio_list", [0.5, 0.5, 0])[0]
                med_ratio = self.prompt_config["enc"].get("ratio_list", [0.5, 0.5, 0])[1]

                context = [-(i + 1) for i in range(int(first_ratio * prompt_len))] + [self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode(d["sentence1"]) + \
                            [self.tokenizer.get_sentinel_id(1)] + self.tokenizer.encode(self.label_word_map[d["label"]]) + \
                            [self.tokenizer.get_sentinel_id(2)] + [-(i + 1) for i in range(int(first_ratio * prompt_len), int((first_ratio + med_ratio) * prompt_len))] + self.tokenizer.encode(d["sentence2"]) + [self.tokenizer.get_sentinel_id(3)]
            else:
                context = [self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode(d["sentence1"]) + \
                            [self.tokenizer.get_sentinel_id(1)] + self.tokenizer.encode(self.label_word_map[d["label"]]) + \
                            [self.tokenizer.get_sentinel_id(2)] + self.tokenizer.encode(d["sentence2"]) + [self.tokenizer.get_sentinel_id(3)]

            target = [1, self.tokenizer.get_sentinel_id(0)] + (self.tokenizer.encode(self.label_word_map[d["label"]]) if not self.do_infer else [self.tokenizer.pad_id])
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


class LCQMCDatasetFromPretrain(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(LCQMCDatasetFromPretrain, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):

        self.label_word_map = ["错误", "正确"]
    
        data = []
        enc_sizes = []
        dec_sizes = []
        
        with open(self.path, "r") as f:
            lines = f.readlines()
        
        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)
            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                context = [-(i + 1) for i in range(prompt_len)] + self.tokenizer.encode(d["sentence1"]) + [self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode(d["sentence2"])
            else:
                context = [39] + self.tokenizer.encode(d["sentence1"]) + [41, 62, 39] + self.tokenizer.encode(d["sentence2"]) + [41, 11, 1348, self.tokenizer.get_sentinel_id(0)]
            
            target = [1, self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode(self.label_word_map[d["label"]])
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


class LCQMCDatasetFromPretrainManTemp(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(LCQMCDatasetFromPretrainManTemp, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):

        self.label_word_map = ["错误", "正确"]
    
        data = []
        enc_sizes = []
        dec_sizes = []
        
        with open(self.path, "r") as f:
            lines = f.readlines()
        
        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)
            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                context = [-(i + 1) for i in range(prompt_len)] + [39] + self.tokenizer.encode(d["sentence1"]) + [41, 11, 1348, 19] + [self.tokenizer.get_sentinel_id(0)] + [39] + self.tokenizer.encode(d["sentence2"]) + [41, 11, 1348]
            else:
                context = [39] + self.tokenizer.encode(d["sentence1"]) + [41, 62, 39] + self.tokenizer.encode(d["sentence2"]) + [41, 11, 1348, self.tokenizer.get_sentinel_id(0)]
            
            target = [1, self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode(self.label_word_map[d["label"]])
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


class LCQMCDatasetFromLM(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(LCQMCDatasetFromLM, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):

        self.label_word_map = ["错误", "正确"]
    
        data = []
        enc_sizes = []
        dec_sizes = []
        
        with open(self.path, "r") as f:
            lines = f.readlines()
        
        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)
            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                context = [-(i + 1) for i in range(prompt_len)] + self.tokenizer.encode(d["sentence1"]) + [18] + self.tokenizer.encode(d["sentence2"]) + [self.tokenizer.get_sentinel_id(0)]
            else:
                context = [39] + self.tokenizer.encode(d["sentence1"]) + [41, 62, 39] + self.tokenizer.encode(d["sentence2"]) + [41, 11, 1348, self.tokenizer.get_sentinel_id(0)]
            
            target = [1, self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode(self.label_word_map[d["label"]])
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