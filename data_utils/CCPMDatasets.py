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

class CPM2Dataset(Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        self.args = args
        self.tokenizer = tokenizer
        self.ratio = ratio
        self.path = path
        self.pad_id = tokenizer.pad_id
        self.add_target_post=add_target_post
        self.split = split
        self.do_infer = do_infer
        self.idx = 0
        self.prompt_config = prompt_config
        if cache_path is not None:
            cache_path = os.path.join(cache_path, "cache_{}_{}.pkl".format(path.replace("/", "_"), ratio))
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    self.data, self.max_enc_len, self.max_dec_len = pickle.load(f)
            else:
                self.data, self.max_enc_len, self.max_dec_len = self.process_data()
                with open(cache_path, "wb") as f:
                    pickle.dump((self.data, self.max_enc_len, self.max_dec_len), f)
        else:
            self.data, self.max_enc_len, self.max_dec_len = self.process_data()

        if num > 0:
            self.data = self.data[:num]

        # if prompt_config is not None:
        #     self.data, self.max_enc_len, self.max_dec_len = self.add_prompt_ids(self.data, self.max_enc_len, self.max_dec_len)

        if do_infer:
            total_eval_batch_size = mpu.get_data_parallel_world_size() * args.eval_batch_size
            total_data_num = math.ceil(len(self.data) / total_eval_batch_size) * total_eval_batch_size
            while len(self.data) < total_data_num:
                tmp = self.data[0].copy()
                tmp["idx"] = -1
                self.data.append(tmp)

        print_str = "Path: {} | Ratio:{} | Max enc len: {} | Max dec len: {} | Data num: {}".format(path, ratio, self.max_enc_len, self.max_dec_len, len(self.data))
        print_rank_0(print_str)
        save_rank_0(args, print_str)

    def process_data(self):
        raise NotImplementedError

    def add_prompt_ids(self, data, max_enc_len, max_dec_len):
        enc_prompt_ids = [i for i in range(self.prompt_config["enc"]["prompt_len"])]
        dec_prompt_ids = [i for i in range(self.prompt_config["dec"]["prompt_len"])]
        pad_ids = [self.tokenizer.pad_id for _ in range(self.prompt_config["dec"]["prompt_len"])]

        for d in data:
            d["enc_input_ids"] = enc_prompt_ids + d["enc_input_ids"]
            d["dec_input_ids"] = dec_prompt_ids + d["dec_input_ids"]
            d["label_ids"] = pad_ids + d["label_ids"]

        max_enc_len += self.prompt_config["enc"]["prompt_len"]
        max_dec_len += self.prompt_config["dec"]["prompt_len"]

        return data, max_enc_len, max_dec_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate(self, samples):
        bs = len(samples)
        model_data = {
            "enc_input_ids": torch.ones(bs, self.max_enc_len, dtype=torch.long) * self.pad_id,
            "enc_attention_mask": torch.zeros(bs, 1, self.max_enc_len, self.max_enc_len),
            "dec_attention_mask": torch.zeros(bs, 1, self.max_dec_len, self.max_dec_len),
            "cross_attention_mask": torch.zeros(bs, 1, self.max_dec_len, self.max_enc_len),
            "dec_input_ids": torch.ones(bs, self.max_dec_len, dtype=torch.long) * self.pad_id
        }
        if not self.do_infer:
            no_model_data = {
                "idx": torch.zeros(bs, dtype=torch.long),
                "labels": torch.ones(bs, self.max_dec_len, dtype=torch.long) * self.pad_id,
                "loss_mask": torch.zeros(bs, self.max_dec_len)
            }
        else:
            no_model_data = {
                "idx": torch.zeros(bs, dtype=torch.long),
            }

        for i, samp in enumerate(samples):
            enc_len, dec_len = len(samp["enc_input_ids"]), len(samp["dec_input_ids"])
            model_data["enc_input_ids"][i][:enc_len] = torch.tensor(samp["enc_input_ids"], dtype=torch.long)
            model_data["dec_input_ids"][i][:dec_len] = torch.tensor(samp["dec_input_ids"], dtype=torch.long)
            model_data["enc_attention_mask"][i][0, :enc_len, :enc_len] = samp.get("enc_attention_mask", 1.0)
            model_data["dec_attention_mask"][i][0, :dec_len, :dec_len] = torch.tril(torch.ones(dec_len, dec_len))
            model_data["cross_attention_mask"][i][0, :dec_len, :enc_len] = samp.get("cross_attention_mask", 1.0)
            no_model_data["idx"][i] = samp["idx"]
            if not self.do_infer:
                no_model_data["labels"][i][:len(samp["label_ids"])] = torch.tensor(samp["label_ids"], dtype=torch.long)
                if self.prompt_config is not None:
                    no_model_data["loss_mask"][i][self.prompt_config["dec"]["prompt_len"]:len(samp["label_ids"])] = 1.0
                else:
                    no_model_data["loss_mask"][i][:len(samp["label_ids"])] = 1.0

        if self.args.fp16:
            model_data["enc_attention_mask"] = model_data["enc_attention_mask"].half()
            model_data["dec_attention_mask"] = model_data["dec_attention_mask"].half()
            model_data["cross_attention_mask"] = model_data["cross_attention_mask"].half()

        return model_data, no_model_data


class PoemMCDataset(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(PoemMCDataset, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)
    
    def process_data(self):
        data = []
        enc_sizes, dec_sizes = [], []
        
        number_map = [
            50, # 一
            230,
            156,
            349,
        ]

        with open(self.path, "r") as f:
            lines = f.readlines()

        for d in lines[:int(self.ratio * len(lines))]:
            d = json.loads(d)
            context_ids = self.tokenizer.encode(d["translation"])
            choice_ids = []

            if self.prompt_config:
                k = 0
                for i, choice in enumerate(d["choices"]):
                    choice_ids.extend([-(k + 1), -(k + 2), -(k + 3), number_map[i], 20] + self.tokenizer.encode(choice) + [18])
                    k += 3
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                first_ratio = self.prompt_config["enc"].get("ratio_list", [0.5, 0.5, 0, 0])[0]
                med_ratio = self.prompt_config["enc"].get("ratio_list", [0.5, 0.5, 0, 0])[1]
                med_token = [] if med_ratio > 0.0 else [18]
                enc_input_ids = [-(i + 1) for i in range(k, int(first_ratio * prompt_len))] + \
                                choice_ids + med_token + [-(i + 1) for i in range(int((first_ratio) * prompt_len), int((first_ratio + med_ratio) * prompt_len))] + \
                                context_ids + [-(i + 1) for i in range(int((first_ratio + med_ratio) * prompt_len), prompt_len)]
            else:
                for i, choice in enumerate(d["choices"]):
                    choice_ids.extend([number_map[i], 20] + self.tokenizer.encode(choice) + [18])
                enc_input_ids = self.tokenizer.encode("选项：") + choice_ids + self.tokenizer.encode("文章：") + context_ids

            target = [1, self.tokenizer.get_sentinel_id(0)] + ([number_map[d["answer"]]] if not self.do_infer else [self.tokenizer.pad_id])
            if self.add_target_post:
                target += [self.tokenizer.get_sentinel_id(1)]
            data.append({
                "idx": self.idx,
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


class PoemMCDatasetTemplateGen(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(PoemMCDatasetTemplateGen, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)
    
    def process_data(self):
        data = []
        enc_sizes, dec_sizes = [], []
        
        number_map = [
            50, # 一
            230,
            156,
            349,
        ]

        with open(self.path, "r") as f:
            lines = f.readlines()

        for d in lines[:int(self.ratio * len(lines))]:
            d = json.loads(d)
            context_ids = self.tokenizer.encode(d["translation"])
            choice_ids = []

            if self.prompt_config:
                for i, choice in enumerate(d["choices"]):
                    choice_ids.extend([number_map[i], 20] + self.tokenizer.encode(choice) + [18])
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                first_ratio = 1
                enc_input_ids = [-(i + 1) for i in range(int(first_ratio * prompt_len))] + [12] + context_ids + [14] + choice_ids + [15] + [self.tokenizer.get_sentinel_id(0)] + [20]
            else:
                for i, choice in enumerate(d["choices"]):
                    choice_ids.extend([number_map[i], 20] + self.tokenizer.encode(choice) + [18])
                enc_input_ids = self.tokenizer.encode("选项：") + choice_ids + self.tokenizer.encode("文章：") + context_ids

            target = [1, self.tokenizer.get_sentinel_id(0)] + ([number_map[d["answer"]]] if not self.do_infer else [self.tokenizer.pad_id])
            if self.add_target_post:
                target += [self.tokenizer.get_sentinel_id(1)]
            data.append({
                "idx": self.idx,
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


class PoemMCDatasetGenTemp(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(PoemMCDatasetGenTemp, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)
    
    def process_data(self):
        data = []
        enc_sizes, dec_sizes = [], []
        
        number_map = [
            50, # 一
            230,
            156,
            349,
        ]

        with open(self.path, "r") as f:
            lines = f.readlines()

        for d in lines[:int(self.ratio * len(lines))]:
            d = json.loads(d)
            context_ids = self.tokenizer.encode(d["translation"])
            choice_ids = []

            if self.prompt_config:
                for i, choice in enumerate(d["choices"]):
                    choice_ids.extend([number_map[i], 20] + self.tokenizer.encode(choice) + [18])
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                first_ratio = 1
                enc_input_ids = [-(i + 1) for i in range(int(first_ratio * prompt_len))] + \
                                [self.tokenizer.get_sentinel_id(0)] + context_ids + \
                                [self.tokenizer.get_sentinel_id(1)] + choice_ids + \
                                [self.tokenizer.get_sentinel_id(2)] + [number_map[d["answer"]]] + [self.tokenizer.get_sentinel_id(3)]
            else:
                for i, choice in enumerate(d["choices"]):
                    choice_ids.extend([number_map[i], 20] + self.tokenizer.encode(choice) + [18])
                enc_input_ids = [self.tokenizer.get_sentinel_id(0)] + context_ids + \
                                [self.tokenizer.get_sentinel_id(1)] + choice_ids + \
                                [self.tokenizer.get_sentinel_id(2)] + [number_map[d["answer"]]] + [self.tokenizer.get_sentinel_id(3)]

            target = [1, self.tokenizer.get_sentinel_id(0)] + ([number_map[d["answer"]]] if not self.do_infer else [self.tokenizer.pad_id])
            if self.add_target_post:
                target += [self.tokenizer.get_sentinel_id(1)]
            data.append({
                "idx": self.idx,
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


class PoemMCDatasetFromPretrain(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(PoemMCDatasetFromPretrain, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)
    
    def process_data(self):
        data = []
        enc_sizes, dec_sizes = [], []
        
        number_map = [
            50, # 一
            230,
            156,
            349,
        ]

        with open(self.path, "r") as f:
            lines = f.readlines()

        for d in lines[:int(self.ratio * len(lines))]:
            d = json.loads(d)
            context_ids = self.tokenizer.encode(d["translation"])
            choice_ids = []

            if self.prompt_config:
                for i, choice in enumerate(d["choices"]):
                    choice_ids.extend([number_map[i], 20] + self.tokenizer.encode(choice) + [18])
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                enc_input_ids = [-(i + 1) for i in range(prompt_len)] + context_ids + [self.tokenizer.get_sentinel_id(0)] + choice_ids
            else:
                for i, choice in enumerate(d["choices"]):
                    choice_ids.extend([number_map[i], 20] + self.tokenizer.encode(choice) + [18])
                enc_input_ids = self.tokenizer.encode("选项：") + choice_ids + self.tokenizer.encode("文章：") + context_ids

            target = [1, self.tokenizer.get_sentinel_id(0)] + ([number_map[d["answer"]]] if not self.do_infer else [self.tokenizer.pad_id])
            if self.add_target_post:
                target += [self.tokenizer.get_sentinel_id(1)]
            data.append({
                "idx": self.idx,
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


class PoemMCDatasetFromPretrainMan(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(PoemMCDatasetFromPretrainMan, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)
    
    def process_data(self):
        data = []
        enc_sizes, dec_sizes = [], []
        
        number_map = [
            50, # 一
            230,
            156,
            349,
        ]

        with open(self.path, "r") as f:
            lines = f.readlines()

        for d in lines[:int(self.ratio * len(lines))]:
            d = json.loads(d)
            context_ids = self.tokenizer.encode(d["translation"])
            choice_ids = []

            if self.prompt_config:
                for i, choice in enumerate(d["choices"]):
                    choice_ids.extend([number_map[i], 20] + self.tokenizer.encode(choice) + [18])
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                enc_input_ids = [-(i + 1) for i in range(prompt_len)] + self.tokenizer.encode("文章：") + context_ids + [19] + [self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode("选项：") + choice_ids
            else:
                for i, choice in enumerate(d["choices"]):
                    choice_ids.extend([number_map[i], 20] + self.tokenizer.encode(choice) + [18])
                enc_input_ids = self.tokenizer.encode("选项：") + choice_ids + self.tokenizer.encode("文章：") + context_ids

            target = [1, self.tokenizer.get_sentinel_id(0)] + ([number_map[d["answer"]]] if not self.do_infer else [self.tokenizer.pad_id])
            if self.add_target_post:
                target += [self.tokenizer.get_sentinel_id(1)]
            data.append({
                "idx": self.idx,
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


class PoemMCDatasetFromLM(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(PoemMCDatasetFromLM, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)
    
    def process_data(self):
        data = []
        enc_sizes, dec_sizes = [], []
        
        number_map = [
            50, # 一
            230,
            156,
            349,
        ]

        with open(self.path, "r") as f:
            lines = f.readlines()

        for d in lines[:int(self.ratio * len(lines))]:
            d = json.loads(d)
            context_ids = self.tokenizer.encode(d["translation"])
            choice_ids = []

            if self.prompt_config:
                for i, choice in enumerate(d["choices"]):
                    choice_ids.extend([number_map[i], 20] + self.tokenizer.encode(choice) + [18])
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                enc_input_ids = [-(i + 1) for i in range(prompt_len)] + context_ids + [18] + choice_ids + [self.tokenizer.get_sentinel_id(0)]
            else:
                for i, choice in enumerate(d["choices"]):
                    choice_ids.extend([number_map[i], 20] + self.tokenizer.encode(choice) + [18])
                enc_input_ids = self.tokenizer.encode("选项：") + choice_ids + self.tokenizer.encode("文章：") + context_ids

            target = [1, self.tokenizer.get_sentinel_id(0)] + ([number_map[d["answer"]]] if not self.do_infer else [self.tokenizer.pad_id])
            if self.add_target_post:
                target += [self.tokenizer.get_sentinel_id(1)]
            data.append({
                "idx": self.idx,
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