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


class C3Dataset(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(C3Dataset, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)
    
    def process_data(self):
        data = []
        enc_sizes, dec_sizes = [], []
        
        number_map = [
            50, # 一
            230,
            156,
            349,
            443,
            803,
            950,
            1031 # 八
        ]

        with open(self.path, "r") as f:
            jobj = json.load(f)

        for d in jobj[:int(self.ratio * len(jobj))]:
            context_ids = self.tokenizer.encode(d[0][0])
            for qa in d[1]:
                question_ids = self.tokenizer.encode(qa["question"])
                choice_ids = []

                if self.prompt_config:
                    k = 0
                    for i, choice in enumerate(qa["choice"]):
                        choice_ids.extend([-(k + 1), -(k + 2), -(k + 3), number_map[i], 20] + self.tokenizer.encode(choice) + [18])
                        k += 3
                    prompt_len = self.prompt_config["enc"]["prompt_len"]
                    first_ratio = self.prompt_config["enc"].get("ratio_list", [0.5, 0.5, 0, 0])[0]
                    med_ratio = self.prompt_config["enc"].get("ratio_list", [0.5, 0.5, 0, 0])[1]
                    med2_ratio = self.prompt_config["enc"].get("ratio_list", [0.5, 0.5, 0, 0])[2]
                    med_token = [] if med_ratio > 0.0 else [18]
                    med2_token = [] if med2_ratio > 0.0 else [18]
                    enc_input_ids = [-(i + 1) for i in range(k, int(first_ratio * prompt_len))] + \
                                    question_ids + med_token + [-(i + 1) for i in range(int(first_ratio * prompt_len), int((first_ratio + med_ratio) * prompt_len))] + \
                                    choice_ids + med2_token + [-(i + 1) for i in range(int((first_ratio + med_ratio) * prompt_len), int((first_ratio + med_ratio + med2_ratio) * prompt_len))] + \
                                    context_ids + [-(i + 1) for i in range(int((first_ratio + med_ratio + med2_ratio) * prompt_len), prompt_len)]
                else:
                    for i, choice in enumerate(qa["choice"]):
                        choice_ids.extend([number_map[i], 20] + self.tokenizer.encode(choice) + [18])
                    enc_input_ids = self.tokenizer.encode("问题：") + question_ids + self.tokenizer.encode("选项：") + choice_ids + self.tokenizer.encode("文章：") + context_ids
                enc_input_ids = enc_input_ids[:512]
                target = [1, self.tokenizer.get_sentinel_id(0)] + ([number_map[qa["choice"].index(qa["answer"])]] if not self.do_infer else [self.tokenizer.pad_id])                
                if self.add_target_post:
                    target += [self.tokenizer.get_sentinel_id(1)]
                data.append({
                    "idx": qa["id"] if self.do_infer else self.idx,
                    "enc_input_ids": enc_input_ids,
                    "dec_input_ids": target[:-1],
                    "label_ids": target[1:],
                })

                if self.prompt_config:
                    if self.prompt_config.get("mask_prompt2token", False):
                        p_mask = (torch.tensor(enc_input_ids) < 0).unsqueeze(1).repeat(1, len(enc_input_ids))
                        enc_attention_mask = (p_mask & p_mask.T | (~p_mask)).float()
                        data[-1].update({
                            "enc_attention_mask": enc_attention_mask 
                        })
                    if self.prompt_config.get("mask_token2prompt", False):
                        p_mask = (torch.tensor(enc_input_ids) < 0).unsqueeze(1).repeat(1, len(enc_input_ids))
                        enc_attention_mask = (p_mask & p_mask.T | (~p_mask)).T.float()
                        # cross_attention_mask = (torch.tensor(context) >= 0).unsqueeze(1).repeat(1, len(target) - 1).T
                        data[-1].update({
                            "enc_attention_mask": enc_attention_mask,
                            # "cross_attention_mask": cross_attention_mask
                        })

                enc_sizes.append(len(enc_input_ids))
                dec_sizes.append(len(target) - 1)
                self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class C3DatasetRaw(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(C3DatasetRaw, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)
    
    def process_data(self):
        data = []
        enc_sizes, dec_sizes = [], []
        
        number_map = [
            50, # 一
            230,
            156,
            349,
            443,
            803,
            950,
            1031 # 八
        ]

        with open(self.path, "r") as f:
            jobj = json.load(f)

        for d in jobj[:int(self.ratio * len(jobj))]:
            context_ids = self.tokenizer.encode(d[0][0])
            for qa in d[1]:
                question_ids = self.tokenizer.encode(qa["question"])
                choice_ids = []

                if self.prompt_config:
                    for i, choice in enumerate(qa["choice"]):
                        choice_ids.extend([number_map[i], 20] + self.tokenizer.encode(choice) + [18])
                    prompt_len = self.prompt_config["enc"]["prompt_len"]
                    context_len = 512 - len(choice_ids) - len(question_ids) - 1 - 2 - prompt_len
                    context_ids = context_ids[:context_len]
                    enc_input_ids = [-(i + 1) for i in range(prompt_len)] + context_ids + self.tokenizer.encode("问题：") + question_ids + [self.tokenizer.get_sentinel_id(0)] + choice_ids
                else:
                    for i, choice in enumerate(qa["choice"]):
                        choice_ids.extend([number_map[i], 20] + self.tokenizer.encode(choice) + [18])
                    enc_input_ids = self.tokenizer.encode("问题：") + question_ids + self.tokenizer.encode("选项：") + choice_ids + self.tokenizer.encode("文章：") + context_ids
                target = [1, self.tokenizer.get_sentinel_id(0)] + ([number_map[qa["choice"].index(qa["answer"])]] if not self.do_infer else [self.tokenizer.pad_id])                
                if self.add_target_post:
                    target += [self.tokenizer.get_sentinel_id(1)]
                data.append({
                    "idx": qa["id"] if self.do_infer else self.idx,
                    "enc_input_ids": enc_input_ids,
                    "dec_input_ids": target[:-1],
                    "label_ids": target[1:],
                })

                if self.prompt_config:
                    if self.prompt_config.get("mask_prompt2token", False):
                        p_mask = (torch.tensor(enc_input_ids) < 0).unsqueeze(1).repeat(1, len(enc_input_ids))
                        enc_attention_mask = (p_mask & p_mask.T | (~p_mask)).float()
                        data[-1].update({
                            "enc_attention_mask": enc_attention_mask 
                        })
                    if self.prompt_config.get("mask_token2prompt", False):
                        p_mask = (torch.tensor(enc_input_ids) < 0).unsqueeze(1).repeat(1, len(enc_input_ids))
                        enc_attention_mask = (p_mask & p_mask.T | (~p_mask)).T.float()
                        # cross_attention_mask = (torch.tensor(context) >= 0).unsqueeze(1).repeat(1, len(target) - 1).T
                        data[-1].update({
                            "enc_attention_mask": enc_attention_mask,
                            # "cross_attention_mask": cross_attention_mask
                        })

                enc_sizes.append(len(enc_input_ids))
                dec_sizes.append(len(target) - 1)
                self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class C3DatasetTemplate(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(C3DatasetTemplate, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)
    
    def process_data(self):
        data = []
        enc_sizes, dec_sizes = [], []
        
        number_map = [
            50, # 一
            230,
            156,
            349,
            443,
            803,
            950,
            1031 # 八
        ]

        with open(self.path, "r") as f:
            jobj = json.load(f)

        for d in jobj[:int(self.ratio * len(jobj))]:
            context_ids = self.tokenizer.encode(d[0][0])
            for qa in d[1]:
                question_ids = self.tokenizer.encode(qa["question"])
                choice_ids = []

                if self.prompt_config:
                    for i, choice in enumerate(qa["choice"]):
                        choice_ids.extend([number_map[i], 20] + self.tokenizer.encode(choice) + [18])
                    prompt_len = self.prompt_config["enc"]["prompt_len"]
                    context_len = 512 - len(choice_ids) - len(question_ids) - 2 - 1 - 2 - 1 - 2 - prompt_len
                    context_ids = context_ids[:context_len]
                    enc_input_ids = [-(i + 1) for i in range(prompt_len)] + self.tokenizer.encode("文章：") + context_ids + self.tokenizer.encode("问题：") + question_ids + [19] + [self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode("选项：") + choice_ids
                else:
                    for i, choice in enumerate(qa["choice"]):
                        choice_ids.extend([number_map[i], 20] + self.tokenizer.encode(choice) + [18])
                    enc_input_ids = self.tokenizer.encode("问题：") + question_ids + self.tokenizer.encode("选项：") + choice_ids + self.tokenizer.encode("文章：") + context_ids
                target = [1, self.tokenizer.get_sentinel_id(0)] + ([number_map[qa["choice"].index(qa["answer"])]] if not self.do_infer else [self.tokenizer.pad_id])                
                if self.add_target_post:
                    target += [self.tokenizer.get_sentinel_id(1)]
                data.append({
                    "idx": qa["id"] if self.do_infer else self.idx,
                    "enc_input_ids": enc_input_ids,
                    "dec_input_ids": target[:-1],
                    "label_ids": target[1:],
                })

                if self.prompt_config:
                    if self.prompt_config.get("mask_prompt2token", False):
                        p_mask = (torch.tensor(enc_input_ids) < 0).unsqueeze(1).repeat(1, len(enc_input_ids))
                        enc_attention_mask = (p_mask & p_mask.T | (~p_mask)).float()
                        data[-1].update({
                            "enc_attention_mask": enc_attention_mask 
                        })
                    if self.prompt_config.get("mask_token2prompt", False):
                        p_mask = (torch.tensor(enc_input_ids) < 0).unsqueeze(1).repeat(1, len(enc_input_ids))
                        enc_attention_mask = (p_mask & p_mask.T | (~p_mask)).T.float()
                        # cross_attention_mask = (torch.tensor(context) >= 0).unsqueeze(1).repeat(1, len(target) - 1).T
                        data[-1].update({
                            "enc_attention_mask": enc_attention_mask,
                            # "cross_attention_mask": cross_attention_mask
                        })

                enc_sizes.append(len(enc_input_ids))
                dec_sizes.append(len(target) - 1)
                self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class C3DatasetTemplateGen(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(C3DatasetTemplateGen, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)
    
    def process_data(self):
        data = []
        enc_sizes, dec_sizes = [], []
        
        number_map = [
            50, # 一
            230,
            156,
            349,
            443,
            803,
            950,
            1031 # 八
        ]

        with open(self.path, "r") as f:
            jobj = json.load(f)

        for d in jobj[:int(self.ratio * len(jobj))]:
            context_ids = self.tokenizer.encode(d[0][0])
            for qa in d[1]:
                question_ids = self.tokenizer.encode(qa["question"])
                choice_ids = []

                if self.prompt_config:
                    for i, choice in enumerate(qa["choice"]):
                        choice_ids.extend([number_map[i], 20] + self.tokenizer.encode(choice) + [18])
                    prompt_len = self.prompt_config["enc"]["prompt_len"]
                    context_len = 512 - len(choice_ids) - len(question_ids) - 1 - 2 - 1 - 1 - 3 - 1 - prompt_len
                    context_ids = context_ids[:context_len]
                    enc_input_ids = [-(i + 1) for i in range(prompt_len)] + [12] + context_ids + [34, 49] + question_ids + [26] + [self.tokenizer.get_sentinel_id(0)] + [75, 276, 17] + choice_ids + [14]
                else:
                    for i, choice in enumerate(qa["choice"]):
                        choice_ids.extend([number_map[i], 20] + self.tokenizer.encode(choice) + [18])
                    enc_input_ids = self.tokenizer.encode("问题：") + question_ids + self.tokenizer.encode("选项：") + choice_ids + self.tokenizer.encode("文章：") + context_ids
                    enc_input_ids = enc_input_ids[:512]

                target = [1, self.tokenizer.get_sentinel_id(0)] + ([number_map[qa["choice"].index(qa["answer"])]] if not self.do_infer else [self.tokenizer.pad_id])                
                if self.add_target_post:
                    target += [self.tokenizer.get_sentinel_id(1)]
                data.append({
                    "idx": qa["id"] if self.do_infer else self.idx,
                    "enc_input_ids": enc_input_ids,
                    "dec_input_ids": target[:-1],
                    "label_ids": target[1:],
                })

                if self.prompt_config:
                    if self.prompt_config.get("mask_prompt2token", False):
                        p_mask = (torch.tensor(enc_input_ids) < 0).unsqueeze(1).repeat(1, len(enc_input_ids))
                        enc_attention_mask = (p_mask & p_mask.T | (~p_mask)).float()
                        data[-1].update({
                            "enc_attention_mask": enc_attention_mask 
                        })
                    if self.prompt_config.get("mask_token2prompt", False):
                        p_mask = (torch.tensor(enc_input_ids) < 0).unsqueeze(1).repeat(1, len(enc_input_ids))
                        enc_attention_mask = (p_mask & p_mask.T | (~p_mask)).T.float()
                        # cross_attention_mask = (torch.tensor(context) >= 0).unsqueeze(1).repeat(1, len(target) - 1).T
                        data[-1].update({
                            "enc_attention_mask": enc_attention_mask,
                            # "cross_attention_mask": cross_attention_mask
                        })

                enc_sizes.append(len(enc_input_ids))
                dec_sizes.append(len(target) - 1)
                self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class C3DatasetGenTemp(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(C3DatasetGenTemp, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)
    
    def process_data(self):
        data = []
        enc_sizes, dec_sizes = [], []
        
        number_map = [
            50, # 一
            230,
            156,
            349,
            443,
            803,
            950,
            1031 # 八
        ]

        with open(self.path, "r") as f:
            jobj = json.load(f)

        for d in jobj[:int(self.ratio * len(jobj))]:
            context_ids = self.tokenizer.encode(d[0][0])
            for qa in d[1]:
                question_ids = self.tokenizer.encode(qa["question"])
                choice_ids = []
                for i, choice in enumerate(qa["choice"]):
                    choice_ids.extend([number_map[i], 20] + self.tokenizer.encode(choice) + [18])
                enc_input_ids = [self.tokenizer.get_sentinel_id(0)] + context_ids + \
                                [self.tokenizer.get_sentinel_id(1)] + question_ids + \
                                [self.tokenizer.get_sentinel_id(2)] + [number_map[qa["choice"].index(qa["answer"])]] + \
                                [self.tokenizer.get_sentinel_id(3)] + choice_ids + \
                                [self.tokenizer.get_sentinel_id(4)]
                                
                target = [1, self.tokenizer.get_sentinel_id(0)] + ([number_map[qa["choice"].index(qa["answer"])]] if not self.do_infer else [self.tokenizer.pad_id])                
                if self.add_target_post:
                    target += [self.tokenizer.get_sentinel_id(1)]

                data.append({
                    "idx": qa["id"] if self.do_infer else self.idx,
                    "enc_input_ids": enc_input_ids,
                    "dec_input_ids": target[:-1],
                    "label_ids": target[1:],
                })

                if self.prompt_config:
                    if self.prompt_config.get("mask_prompt2token", False):
                        p_mask = (torch.tensor(enc_input_ids) < 0).unsqueeze(1).repeat(1, len(enc_input_ids))
                        enc_attention_mask = (p_mask & p_mask.T | (~p_mask)).float()
                        data[-1].update({
                            "enc_attention_mask": enc_attention_mask 
                        })
                    if self.prompt_config.get("mask_token2prompt", False):
                        p_mask = (torch.tensor(enc_input_ids) < 0).unsqueeze(1).repeat(1, len(enc_input_ids))
                        enc_attention_mask = (p_mask & p_mask.T | (~p_mask)).T.float()
                        # cross_attention_mask = (torch.tensor(context) >= 0).unsqueeze(1).repeat(1, len(target) - 1).T
                        data[-1].update({
                            "enc_attention_mask": enc_attention_mask,
                            # "cross_attention_mask": cross_attention_mask
                        })

                enc_sizes.append(len(enc_input_ids))
                dec_sizes.append(len(target) - 1)
                self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class C3DatasetFromPretrain(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(C3DatasetFromPretrain, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)
    
    def process_data(self):
        data = []
        enc_sizes, dec_sizes = [], []
        
        number_map = [
            50, # 一
            230,
            156,
            349,
            443,
            803,
            950,
            1031 # 八
        ]

        with open(self.path, "r") as f:
            jobj = json.load(f)

        for d in jobj[:int(self.ratio * len(jobj))]:
            context_ids = self.tokenizer.encode(d[0][0])
            for qa in d[1]:
                question_ids = self.tokenizer.encode(qa["question"])
                choice_ids = []

                if self.prompt_config:
                    for i, choice in enumerate(qa["choice"]):
                        choice_ids.extend([number_map[i], 20] + self.tokenizer.encode(choice) + [18])
                    prompt_len = self.prompt_config["enc"]["prompt_len"]
                    context_len = 512 - len(choice_ids) - len(question_ids) - 1 - 2 - prompt_len
                    context_ids = context_ids[:context_len]
                    enc_input_ids = [-(i + 1) for i in range(prompt_len)] + context_ids + self.tokenizer.encode("问题：") + question_ids + [self.tokenizer.get_sentinel_id(0)] + choice_ids
                else:
                    for i, choice in enumerate(qa["choice"]):
                        choice_ids.extend([number_map[i], 20] + self.tokenizer.encode(choice) + [18])
                    enc_input_ids = self.tokenizer.encode("问题：") + question_ids + self.tokenizer.encode("选项：") + choice_ids + self.tokenizer.encode("文章：") + context_ids
                target = [1, self.tokenizer.get_sentinel_id(0)] + ([number_map[qa["choice"].index(qa["answer"])]] if not self.do_infer else [self.tokenizer.pad_id])                
                if self.add_target_post:
                    target += [self.tokenizer.get_sentinel_id(1)]
                data.append({
                    "idx": qa["id"] if self.do_infer else self.idx,
                    "enc_input_ids": enc_input_ids,
                    "dec_input_ids": target[:-1],
                    "label_ids": target[1:],
                })

                if self.prompt_config:
                    if self.prompt_config.get("mask_prompt2token", False):
                        p_mask = (torch.tensor(enc_input_ids) < 0).unsqueeze(1).repeat(1, len(enc_input_ids))
                        enc_attention_mask = (p_mask & p_mask.T | (~p_mask)).float()
                        data[-1].update({
                            "enc_attention_mask": enc_attention_mask 
                        })
                    if self.prompt_config.get("mask_token2prompt", False):
                        p_mask = (torch.tensor(enc_input_ids) < 0).unsqueeze(1).repeat(1, len(enc_input_ids))
                        enc_attention_mask = (p_mask & p_mask.T | (~p_mask)).T.float()
                        # cross_attention_mask = (torch.tensor(context) >= 0).unsqueeze(1).repeat(1, len(target) - 1).T
                        data[-1].update({
                            "enc_attention_mask": enc_attention_mask,
                            # "cross_attention_mask": cross_attention_mask
                        })

                enc_sizes.append(len(enc_input_ids))
                dec_sizes.append(len(target) - 1)
                self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class C3DatasetFromLM(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(C3DatasetFromLM, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)
    
    def process_data(self):
        data = []
        enc_sizes, dec_sizes = [], []
        
        number_map = [
            50, # 一
            230,
            156,
            349,
            443,
            803,
            950,
            1031 # 八
        ]

        with open(self.path, "r") as f:
            jobj = json.load(f)

        for d in jobj[:int(self.ratio * len(jobj))]:
            context_ids = self.tokenizer.encode(d[0][0])
            for qa in d[1]:
                question_ids = self.tokenizer.encode(qa["question"])
                choice_ids = []

                if self.prompt_config:
                    for i, choice in enumerate(qa["choice"]):
                        choice_ids.extend([number_map[i], 20] + self.tokenizer.encode(choice) + [18])
                    prompt_len = self.prompt_config["enc"]["prompt_len"]
                    context_len = 512 - len(choice_ids) - len(question_ids) - 1 - 4 - prompt_len
                    context_ids = context_ids[:context_len]
                    enc_input_ids = [-(i + 1) for i in range(prompt_len)] + context_ids + self.tokenizer.encode("问题：") + question_ids + self.tokenizer.encode("选项：") + choice_ids + [self.tokenizer.get_sentinel_id(0)]
                else:
                    for i, choice in enumerate(qa["choice"]):
                        choice_ids.extend([number_map[i], 20] + self.tokenizer.encode(choice) + [18])
                    enc_input_ids = self.tokenizer.encode("问题：") + question_ids + self.tokenizer.encode("选项：") + choice_ids + self.tokenizer.encode("文章：") + context_ids
                target = [1, self.tokenizer.get_sentinel_id(0)] + ([number_map[qa["choice"].index(qa["answer"])]] if not self.do_infer else [self.tokenizer.pad_id])                
                if self.add_target_post:
                    target += [self.tokenizer.get_sentinel_id(1)]
                data.append({
                    "idx": qa["id"] if self.do_infer else self.idx,
                    "enc_input_ids": enc_input_ids,
                    "dec_input_ids": target[:-1],
                    "label_ids": target[1:],
                })

                if self.prompt_config:
                    if self.prompt_config.get("mask_prompt2token", False):
                        p_mask = (torch.tensor(enc_input_ids) < 0).unsqueeze(1).repeat(1, len(enc_input_ids))
                        enc_attention_mask = (p_mask & p_mask.T | (~p_mask)).float()
                        data[-1].update({
                            "enc_attention_mask": enc_attention_mask 
                        })
                    if self.prompt_config.get("mask_token2prompt", False):
                        p_mask = (torch.tensor(enc_input_ids) < 0).unsqueeze(1).repeat(1, len(enc_input_ids))
                        enc_attention_mask = (p_mask & p_mask.T | (~p_mask)).T.float()
                        # cross_attention_mask = (torch.tensor(context) >= 0).unsqueeze(1).repeat(1, len(target) - 1).T
                        data[-1].update({
                            "enc_attention_mask": enc_attention_mask,
                            # "cross_attention_mask": cross_attention_mask
                        })

                enc_sizes.append(len(enc_input_ids))
                dec_sizes.append(len(target) - 1)
                self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len

