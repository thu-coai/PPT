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

class CMNLIDataset(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(CMNLIDataset, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):

        self.label_word_map = {
            "contradiction": "矛盾",
            "neutral": "中立",
            "entailment": "相似"
        }
    
        data = []
        enc_sizes = []
        dec_sizes = []
        
        with open(self.path, "r") as f:
            lines = f.readlines()
        
        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)
            if d["label"] == "-":
                continue
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


def cut_to_max_len(prefix, postfix, max_len):
    if len(prefix) + len(postfix) <= max_len:
        return prefix, postfix

    overflow_num = len(prefix)  + len(postfix) - max_len

    overflow_num_prefix = int((len(prefix) / (len(prefix) + len(postfix))) * overflow_num)
    overflow_num_postfix = int((len(postfix) / (len(prefix) + len(postfix))) * overflow_num)
        
    if overflow_num_prefix + overflow_num_postfix < overflow_num:
        if len(prefix) > len(postfix):
            overflow_num_prefix += 1
        else:
            overflow_num_postfix += 1

    assert overflow_num_prefix + overflow_num_postfix >= overflow_num, (overflow_num_prefix, overflow_num_postfix, overflow_num)

    return prefix[overflow_num_prefix:], postfix[:len(postfix) - overflow_num_postfix]


class CHIDDatasetRaw(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(CHIDDatasetRaw, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):
        data = []
        enc_sizes = []
        dec_sizes = []

        with open(self.path, "r") as f:
            lines = f.readlines()
        
        ans_d = None
        if not self.do_infer:
            with open(self.path.replace(".json", "_answer.json"), "r") as f:
                ans_d = json.load(f)

        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)
            for sent in d["content"]:
                samples, tmp_enc_sizes, tmp_dec_sizes = self.process_one_sent(sent, ans_d, d["candidates"])
                data.extend(samples)
                enc_sizes.extend(tmp_enc_sizes)
                dec_sizes.extend(tmp_dec_sizes)

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len

    def process_one_sent(self, sent, answers, cands):
        pattern = re.compile(r"#idiom(\d+)#")
        start = 0
        samples = []
        enc_sizes, dec_sizes = [], []

        number_map = [
            50, # 一
            230,
            156,
            349,
            443,
            803,
            950,
            1031, # 八
            1189, # 九
            1320
        ]

        cands_ids = []
        for i, cand in enumerate(cands):
            cands_ids.extend([number_map[i], 20] + self.tokenizer.encode(cand.strip()) + [18])

        while True:
            m = pattern.search(sent, start)
            if m is None:
                break
            
            prefix = self.tokenizer.encode(re.sub(pattern, "", sent[:m.start()]))
            postfix = self.tokenizer.encode(re.sub(pattern, "", sent[m.end():]))
    
            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                max_len = 512 - len(cands_ids) - 1 - prompt_len - 3 -2
                prefix, postfix = cut_to_max_len(prefix, postfix, max_len)
                context_ids = self.tokenizer.encode("上文：") + prefix + self.tokenizer.encode("下文：") + postfix
                ids = [-(i + 1) for i in range(prompt_len)] + context_ids + [self.tokenizer.get_sentinel_id(0)] + cands_ids    
            else:
                max_len = 512 - len(cands_ids) - 1 - 3 -2
                prefix, postfix = cut_to_max_len(prefix, postfix, max_len)
                context_ids = self.tokenizer.encode("上文：") + prefix + self.tokenizer.encode("下文：") + postfix
                ids = context_ids + [self.tokenizer.get_sentinel_id(0)] + cands_ids

            target = [1, self.tokenizer.get_sentinel_id(0)] + [number_map[answers[m.group()]]]
            if self.add_target_post:
                target += [self.tokenizer.get_sentinel_id(1)]
            
            samples.append({
                "idx": int(m.group(1)) if self.do_infer else self.idx,
                "enc_input_ids": ids,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:]
            })

            enc_sizes.append(len(ids))
            dec_sizes.append(len(target) - 1)
            self.idx += 1
    
            start = m.end()

        return samples, enc_sizes, dec_sizes


class CHIDDatasetFromPretrain(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(CHIDDatasetFromPretrain, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):
        data = []
        enc_sizes = []
        dec_sizes = []

        with open(self.path, "r") as f:
            lines = f.readlines()
        
        ans_d = None
        if not self.do_infer:
            with open(self.path.replace(".json", "_answer.json"), "r") as f:
                ans_d = json.load(f)

        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)
            for sent in d["content"]:
                samples, tmp_enc_sizes, tmp_dec_sizes = self.process_one_sent(sent, ans_d, d["candidates"])
                data.extend(samples)
                enc_sizes.extend(tmp_enc_sizes)
                dec_sizes.extend(tmp_dec_sizes)

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len

    def process_one_sent(self, sent, answers, cands):
        pattern = re.compile(r"#idiom(\d+)#")
        start = 0
        samples = []
        enc_sizes, dec_sizes = [], []

        number_map = [
            50, # 一
            230,
            156,
            349,
            443,
            803,
            950,
            1031, # 八
            1189, # 九
            1320
        ]

        cands_ids = []
        for i, cand in enumerate(cands):
            cands_ids.extend([number_map[i], 20] + self.tokenizer.encode(cand.strip()) + [18])

        while True:
            m = pattern.search(sent, start)
            if m is None:
                break
            
            prefix = self.tokenizer.encode(re.sub(pattern, "", sent[:m.start()]))
            postfix = self.tokenizer.encode(re.sub(pattern, "", sent[m.end():]))
    
            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                max_len = 512 - len(cands_ids) - 1 - prompt_len - 3 -2
                prefix, postfix = cut_to_max_len(prefix, postfix, max_len)
                context_ids = self.tokenizer.encode("上文：") + prefix + self.tokenizer.encode("下文：") + postfix
                ids = [-(i + 1) for i in range(prompt_len)] + context_ids + [self.tokenizer.get_sentinel_id(0)] + cands_ids    
            else:
                max_len = 512 - len(cands_ids) - 1 - 3 -2
                prefix, postfix = cut_to_max_len(prefix, postfix, max_len)
                context_ids = self.tokenizer.encode("上文：") + prefix + self.tokenizer.encode("下文：") + postfix
                ids = context_ids + [self.tokenizer.get_sentinel_id(0)] + cands_ids

            target = [1, self.tokenizer.get_sentinel_id(0)] + [number_map[answers[m.group()]]]
            if self.add_target_post:
                target += [self.tokenizer.get_sentinel_id(1)]
            
            samples.append({
                "idx": int(m.group(1)) if self.do_infer else self.idx,
                "enc_input_ids": ids,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:]
            })

            enc_sizes.append(len(ids))
            dec_sizes.append(len(target) - 1)
            self.idx += 1
    
            start = m.end()

        return samples, enc_sizes, dec_sizes
