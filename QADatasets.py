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

class QADataset(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(QADataset, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)
    
    def process_data(self):
        data = []
        enc_sizes, dec_sizes = [], []
        
        number_map = [
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

        with open(self.path, "r") as f:
            lines = f.readlines()

        for d in lines[:int(self.ratio * len(lines))]:
            d = json.loads(d)
            question_ids = self.tokenizer.encode(d["question"])
            choice_ids = []

            for i, choice in zip(range(17), d["option"]):
                choice_ids.extend([number_map[i], 5] + self.tokenizer.encode(choice) + [5])

            enc_input_ids = question_ids + [58] + choice_ids + [58] + self.tokenizer.encode("The correct one is ") + [self.tokenizer.get_sentinel_id(0)]
            if d["article"] is not None:
                context_ids = self.tokenizer.encode(d["article"])[:340]
                enc_input_ids = context_ids + self.tokenizer.encode("Based on the previous passage, we ask ") + enc_input_ids

            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                enc_input_ids = [-(i + 1) for i in range(prompt_len)] + enc_input_ids

            target = [0, self.tokenizer.get_sentinel_id(0)] + [number_map[ord(d["answer"]) - ord("A")]]
            if self.add_target_post:
                target += [self.tokenizer.get_sentinel_id(1)]
            data.append({
                "idx": self.idx,
                "enc_input_ids": enc_input_ids,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:],
            })

            enc_sizes.append(len(enc_input_ids))
            dec_sizes.append(len(target) - 1)
            self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class QADataset2(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(QADataset2, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)
    
    def process_data(self):
        data = []
        enc_sizes, dec_sizes = [], []
        
        number_map = {
            "A": 188, # A
            "B": 279, # B
            "C": 254, # C
            "D": 308, # D
        }

        with open(self.path, "r") as f:
            lines = f.readlines()

        for d in lines[:int(self.ratio * len(lines))]:
            d = json.loads(d)
            context_ids = self.tokenizer.encode(d["article"])
            question_ids = self.tokenizer.encode(d["question"])
            choice_ids = []

            for i, choice in zip(["A", "B", "C", "D"], d["option"]):
                choice_ids.extend([number_map[i], 5] + self.tokenizer.encode(choice) + [5])

            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                enc_input_ids = [-(i + 1) for i in range(prompt_len)] + context_ids + [117] + question_ids + [self.tokenizer.get_sentinel_id(0)] + choice_ids
            else:
                enc_input_ids = context_ids + self.tokenizer.encode("Based on the previous passage, we ask ") + question_ids + [58] + choice_ids + [58] + self.tokenizer.encode("The correct one is ") + [self.tokenizer.get_sentinel_id(0)]
                # print(self.tokenizer.decode(enc_input_ids))
                # exit(0)

            target = [0, self.tokenizer.get_sentinel_id(0)] + [number_map[d["answer"]]]
            if self.add_target_post:
                target += [self.tokenizer.get_sentinel_id(1)]
            data.append({
                "idx": self.idx,
                "enc_input_ids": enc_input_ids,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:],
            })

            enc_sizes.append(len(enc_input_ids))
            dec_sizes.append(len(target) - 1)
            self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class QADatasetMan(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(QADatasetMan, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)
    
    def process_data(self):
        data = []
        enc_sizes, dec_sizes = [], []
        
        number_map = {
            "A": 188, # A
            "B": 279, # B
            "C": 254, # C
            "D": 308, # D
        }

        with open(self.path, "r") as f:
            lines = f.readlines()

        for d in lines[:int(self.ratio * len(lines))]:
            d = json.loads(d)
            context_ids = self.tokenizer.encode(d["article"])
            question_ids = self.tokenizer.encode(d["question"])
            choice_ids = []

            for i, choice in zip(["A", "B", "C", "D"], d["option"]):
                choice_ids.extend([number_map[i], 5] + self.tokenizer.encode(choice) + [5])

            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                enc_input_ids = [-(i + 1) for i in range(prompt_len)] + context_ids + self.tokenizer.encode("Based on the previous passage, we ask ") + question_ids + [58] + choice_ids + [58] + self.tokenizer.encode("The correct one is ") + [self.tokenizer.get_sentinel_id(0)]
            else:
                enc_input_ids = context_ids + self.tokenizer.encode("Based on the previous passage, we ask ") + question_ids + [58] + choice_ids + [58] + self.tokenizer.encode("The correct one is ") + [self.tokenizer.get_sentinel_id(0)]

            target = [0, self.tokenizer.get_sentinel_id(0)] + [number_map[d["answer"]]]
            if self.add_target_post:
                target += [self.tokenizer.get_sentinel_id(1)]
            data.append({
                "idx": self.idx,
                "enc_input_ids": enc_input_ids,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:],
            })

            enc_sizes.append(len(enc_input_ids))
            dec_sizes.append(len(target) - 1)
            self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class QADatasetGenTemp(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(QADatasetGenTemp, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)
    
    def process_data(self):
        data = []
        enc_sizes, dec_sizes = [], []
        
        number_map = {
            "A": 188, # A
            "B": 279, # B
            "C": 254, # C
            "D": 308, # D
        }

        with open(self.path, "r") as f:
            lines = f.readlines()

        for d in lines[:int(self.ratio * len(lines))]:
            d = json.loads(d)
            context_ids = self.tokenizer.encode(d["article"])
            question_ids = self.tokenizer.encode(d["question"])
            choice_ids = []

            for i, choice in zip(["A", "B", "C", "D"], d["option"]):
                choice_ids.extend([number_map[i], 5] + self.tokenizer.encode(choice) + [5])

            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                enc_input_ids = [-(i + 1) for i in range(prompt_len)] + [self.tokenizer.get_sentinel_id(0)] + \
                                                          context_ids + [self.tokenizer.get_sentinel_id(1)] + \
                                                         question_ids + [self.tokenizer.get_sentinel_id(2)] + \
                                                           choice_ids + [self.tokenizer.get_sentinel_id(3)] + \
                                            [number_map[d["answer"]]] + [self.tokenizer.get_sentinel_id(4)]
            else:
                enc_input_ids = context_ids + self.tokenizer.encode("Based on the previous passage, we ask ") + question_ids + [58] + choice_ids + [58] + self.tokenizer.encode("The correct one is ") + [self.tokenizer.get_sentinel_id(0)]

            target = [0, self.tokenizer.get_sentinel_id(0)] + [number_map[d["answer"]]]
            if self.add_target_post:
                target += [self.tokenizer.get_sentinel_id(1)]
            data.append({
                "idx": self.idx,
                "enc_input_ids": enc_input_ids,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:],
            })

            enc_sizes.append(len(enc_input_ids))
            dec_sizes.append(len(target) - 1)
            self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class RACEMDatasetGenRandom(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(RACEMDatasetGenRandom, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)
    
    def process_data(self):
        data = []
        enc_sizes, dec_sizes = [], []
        
        number_map = {
            "A": 188, # A
            "B": 279, # B
            "C": 254, # C
            "D": 308, # D
        }

        with open(self.path, "r") as f:
            lines = f.readlines()

        for d in lines[:int(self.ratio * len(lines))]:
            d = json.loads(d)
            context_ids = self.tokenizer.encode(d["article"])
            question_ids = self.tokenizer.encode(d["question"])
            choice_ids = []

            for i, choice in zip(["A", "B", "C", "D"], d["option"]):
                choice_ids.extend([number_map[i], 5] + self.tokenizer.encode(choice) + [5])

            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                enc_input_ids = [-(i + 1) for i in range(prompt_len)] + [5] + context_ids + [1300] + question_ids + [3, 5] + choice_ids + [1682] + [self.tokenizer.get_sentinel_id(0)] + [5, 3, 5]
            else:
                enc_input_ids = context_ids + self.tokenizer.encode("Based on the previous passage, we ask ") + question_ids + [58] + choice_ids + [58] + self.tokenizer.encode("The correct one is ") + [self.tokenizer.get_sentinel_id(0)]

            target = [0, self.tokenizer.get_sentinel_id(0)] + [number_map[d["answer"]]]
            if self.add_target_post:
                target += [self.tokenizer.get_sentinel_id(1)]
            data.append({
                "idx": self.idx,
                "enc_input_ids": enc_input_ids,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:],
            })

            enc_sizes.append(len(enc_input_ids))
            dec_sizes.append(len(target) - 1)
            self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class RACEMDatasetGenVocab(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(RACEMDatasetGenVocab, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)
    
    def process_data(self):
        data = []
        enc_sizes, dec_sizes = [], []
        
        number_map = {
            "A": 188, # A
            "B": 279, # B
            "C": 254, # C
            "D": 308, # D
        }

        with open(self.path, "r") as f:
            lines = f.readlines()

        for d in lines[:int(self.ratio * len(lines))]:
            d = json.loads(d)
            context_ids = self.tokenizer.encode(d["article"])
            question_ids = self.tokenizer.encode(d["question"])
            choice_ids = []

            for i, choice in zip(["A", "B", "C", "D"], d["option"]):
                choice_ids.extend([number_map[i], 5] + self.tokenizer.encode(choice) + [5])

            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                enc_input_ids = [-(i + 1) for i in range(prompt_len)] + [15, 7, 159, 5] + context_ids + [1300] + question_ids + [3, 5] + choice_ids + [1682] + [self.tokenizer.get_sentinel_id(0)] + [5]
            else:
                enc_input_ids = context_ids + self.tokenizer.encode("Based on the previous passage, we ask ") + question_ids + [58] + choice_ids + [58] + self.tokenizer.encode("The correct one is ") + [self.tokenizer.get_sentinel_id(0)]

            target = [0, self.tokenizer.get_sentinel_id(0)] + [number_map[d["answer"]]]
            if self.add_target_post:
                target += [self.tokenizer.get_sentinel_id(1)]
            data.append({
                "idx": self.idx,
                "enc_input_ids": enc_input_ids,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:],
            })

            enc_sizes.append(len(enc_input_ids))
            dec_sizes.append(len(target) - 1)
            self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class RACEHDatasetGenRandom(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(RACEHDatasetGenRandom, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)
    
    def process_data(self):
        data = []
        enc_sizes, dec_sizes = [], []
        
        number_map = {
            "A": 188, # A
            "B": 279, # B
            "C": 254, # C
            "D": 308, # D
        }

        with open(self.path, "r") as f:
            lines = f.readlines()

        for d in lines[:int(self.ratio * len(lines))]:
            d = json.loads(d)
            context_ids = self.tokenizer.encode(d["article"])
            question_ids = self.tokenizer.encode(d["question"])
            choice_ids = []

            for i, choice in zip(["A", "B", "C", "D"], d["option"]):
                choice_ids.extend([number_map[i], 5] + self.tokenizer.encode(choice) + [5])

            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                enc_input_ids = [-(i + 1) for i in range(prompt_len)] + [5] + context_ids + [3, 5] + question_ids + [3, 5] + choice_ids + [3, 5] + [self.tokenizer.get_sentinel_id(0)] + [5]
            else:
                enc_input_ids = context_ids + self.tokenizer.encode("Based on the previous passage, we ask ") + question_ids + [58] + choice_ids + [58] + self.tokenizer.encode("The correct one is ") + [self.tokenizer.get_sentinel_id(0)]

            target = [0, self.tokenizer.get_sentinel_id(0)] + [number_map[d["answer"]]]
            if self.add_target_post:
                target += [self.tokenizer.get_sentinel_id(1)]
            data.append({
                "idx": self.idx,
                "enc_input_ids": enc_input_ids,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:],
            })

            enc_sizes.append(len(enc_input_ids))
            dec_sizes.append(len(target) - 1)
            self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class RACEHDatasetGenVocab(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(RACEHDatasetGenVocab, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)
    
    def process_data(self):
        data = []
        enc_sizes, dec_sizes = [], []
        
        number_map = {
            "A": 188, # A
            "B": 279, # B
            "C": 254, # C
            "D": 308, # D
        }

        with open(self.path, "r") as f:
            lines = f.readlines()

        for d in lines[:int(self.ratio * len(lines))]:
            d = json.loads(d)
            context_ids = self.tokenizer.encode(d["article"])
            question_ids = self.tokenizer.encode(d["question"])
            choice_ids = []

            for i, choice in zip(["A", "B", "C", "D"], d["option"]):
                choice_ids.extend([number_map[i], 5] + self.tokenizer.encode(choice) + [5])

            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                enc_input_ids = [-(i + 1) for i in range(prompt_len)] + [15, 7, 159, 5] + context_ids + [3, 5] + question_ids + [3, 5] + choice_ids + [3, 5] + [self.tokenizer.get_sentinel_id(0)] + [5]
            else:
                enc_input_ids = context_ids + self.tokenizer.encode("Based on the previous passage, we ask ") + question_ids + [58] + choice_ids + [58] + self.tokenizer.encode("The correct one is ") + [self.tokenizer.get_sentinel_id(0)]

            target = [0, self.tokenizer.get_sentinel_id(0)] + [number_map[d["answer"]]]
            if self.add_target_post:
                target += [self.tokenizer.get_sentinel_id(1)]
            data.append({
                "idx": self.idx,
                "enc_input_ids": enc_input_ids,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:],
            })

            enc_sizes.append(len(enc_input_ids))
            dec_sizes.append(len(target) - 1)
            self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class QADatasetFromPretrain(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(QADatasetFromPretrain, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)
    
    def process_data(self):
        data = []
        enc_sizes, dec_sizes = [], []
        
        number_map = {
            "A": 188, # A
            "B": 279, # B
            "C": 254, # C
            "D": 308, # D
        }

        with open(self.path, "r") as f:
            lines = f.readlines()

        for d in lines[:int(self.ratio * len(lines))]:
            d = json.loads(d)
            context_ids = self.tokenizer.encode(d["article"])
            question_ids = self.tokenizer.encode(d["question"])
            choice_ids = []

            for i, choice in zip(["A", "B", "C", "D"], d["option"]):
                choice_ids.extend([number_map[i], 5] + self.tokenizer.encode(choice) + [5])

            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                enc_input_ids = [-(i + 1) for i in range(prompt_len)] + context_ids + [117] + question_ids + [58] + choice_ids + [58] + self.tokenizer.encode("The correct one is ") + [self.tokenizer.get_sentinel_id(0)]
            else:
                enc_input_ids = context_ids + self.tokenizer.encode("Based on the previous passage, we ask ") + question_ids + [58] + choice_ids + [58] + self.tokenizer.encode("The correct one is ") + [self.tokenizer.get_sentinel_id(0)]
                # print(self.tokenizer.decode(enc_input_ids))
                # exit(0)

            target = [0, self.tokenizer.get_sentinel_id(0)] + [number_map[d["answer"]]]
            if self.add_target_post:
                target += [self.tokenizer.get_sentinel_id(1)]
            data.append({
                "idx": self.idx,
                "enc_input_ids": enc_input_ids,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:],
            })

            enc_sizes.append(len(enc_input_ids))
            dec_sizes.append(len(target) - 1)
            self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class QADatasetFromLM(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(QADatasetFromLM, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)
    
    def process_data(self):
        data = []
        enc_sizes, dec_sizes = [], []
        
        number_map = {
            "A": 188, # A
            "B": 279, # B
            "C": 254, # C
            "D": 308, # D
        }

        with open(self.path, "r") as f:
            lines = f.readlines()

        for d in lines[:int(self.ratio * len(lines))]:
            d = json.loads(d)
            context_ids = self.tokenizer.encode(d["article"])
            question_ids = self.tokenizer.encode(d["question"])
            choice_ids = []

            for i, choice in zip(["A", "B", "C", "D"], d["option"]):
                choice_ids.extend([number_map[i], 5] + self.tokenizer.encode(choice) + [5])

            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                enc_input_ids = [-(i + 1) for i in range(prompt_len)] + self.tokenizer.encode("options：") + choice_ids + self.tokenizer.encode("passage: ") + context_ids + self.tokenizer.encode("question: ") + question_ids
            else:
                enc_input_ids = context_ids + self.tokenizer.encode("Based on the previous passage, we ask ") + question_ids + [58] + choice_ids + [58] + self.tokenizer.encode("The correct one is ") + [self.tokenizer.get_sentinel_id(0)]
                # print(self.tokenizer.decode(enc_input_ids))
                # exit(0)

            target = [0] + [number_map[d["answer"]]]

            data.append({
                "idx": self.idx,
                "enc_input_ids": enc_input_ids,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:],
            })

            enc_sizes.append(len(enc_input_ids))
            dec_sizes.append(len(target) - 1)
            self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class QADatasetFromPretrainLabel2(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(QADatasetFromPretrainLabel2, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)
    
    def process_data(self):
        data = []
        enc_sizes, dec_sizes = [], []
        
        number_map = {
            "A": 5411, # A
            "B": 4416, # B
            "C": 5787, # C
            "D": 7984, # D
        }

        with open(self.path, "r") as f:
            lines = f.readlines()

        for d in lines[:int(self.ratio * len(lines))]:
            d = json.loads(d)
            context_ids = self.tokenizer.encode(d["article"])
            question_ids = self.tokenizer.encode(d["question"])
            choice_ids = []

            for i, choice in zip(["A", "B", "C", "D"], d["option"]):
                choice_ids.extend([number_map[i]] + self.tokenizer.encode(choice) + [5])

            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                enc_input_ids = [-(i + 1) for i in range(prompt_len)] + context_ids + [117] + question_ids + [58] + choice_ids + [58] + self.tokenizer.encode("The correct one is ") + [self.tokenizer.get_sentinel_id(0)]
            else:
                pass

            target = [0, self.tokenizer.get_sentinel_id(0)] + [number_map[d["answer"]]]
            if self.add_target_post:
                target += [self.tokenizer.get_sentinel_id(1)]
            data.append({
                "idx": self.idx,
                "enc_input_ids": enc_input_ids,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:],
            })

            enc_sizes.append(len(enc_input_ids))
            dec_sizes.append(len(target) - 1)
            self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class QADatasetFromPretrainLabel3(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(QADatasetFromPretrainLabel3, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)
    
    def process_data(self):
        data = []
        enc_sizes, dec_sizes = [], []
        
        number_map = {
            "A": 279, # B
            "B": 308, # D
            "C": 254, # C
            "D": 188, # A
        }

        with open(self.path, "r") as f:
            lines = f.readlines()

        for d in lines[:int(self.ratio * len(lines))]:
            d = json.loads(d)
            context_ids = self.tokenizer.encode(d["article"])
            question_ids = self.tokenizer.encode(d["question"])
            choice_ids = []

            for i, choice in zip(["A", "B", "C", "D"], d["option"]):
                choice_ids.extend([number_map[i]] + self.tokenizer.encode(choice) + [5])

            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                enc_input_ids = [-(i + 1) for i in range(prompt_len)] + context_ids + [117] + question_ids + [58] + choice_ids + [58] + self.tokenizer.encode("The correct one is ") + [self.tokenizer.get_sentinel_id(0)]
            else:
                pass

            target = [0, self.tokenizer.get_sentinel_id(0)] + [number_map[d["answer"]]]
            if self.add_target_post:
                target += [self.tokenizer.get_sentinel_id(1)]
            data.append({
                "idx": self.idx,
                "enc_input_ids": enc_input_ids,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:],
            })

            enc_sizes.append(len(enc_input_ids))
            dec_sizes.append(len(target) - 1)
            self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class QADatasetFromPretrainLabel4(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(QADatasetFromPretrainLabel4, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)
    
    def process_data(self):
        data = []
        enc_sizes, dec_sizes = [], []
        
        number_map = {
            "A": 10169, # 2.
            "B": 2138, # 4.
            "C": 9414, # 3.
            "D": 6779, # 1.
        }

        with open(self.path, "r") as f:
            lines = f.readlines()

        for d in lines[:int(self.ratio * len(lines))]:
            d = json.loads(d)
            context_ids = self.tokenizer.encode(d["article"])
            question_ids = self.tokenizer.encode(d["question"])
            choice_ids = []

            for i, choice in zip(["A", "B", "C", "D"], d["option"]):
                choice_ids.extend([number_map[i]] + self.tokenizer.encode(choice) + [5])

            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                enc_input_ids = [-(i + 1) for i in range(prompt_len)] + context_ids + [117] + question_ids + [58] + choice_ids + [58] + self.tokenizer.encode("The correct one is ") + [self.tokenizer.get_sentinel_id(0)]
            else:
                pass

            target = [0, self.tokenizer.get_sentinel_id(0)] + [number_map[d["answer"]]]
            if self.add_target_post:
                target += [self.tokenizer.get_sentinel_id(1)]
            data.append({
                "idx": self.idx,
                "enc_input_ids": enc_input_ids,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:],
            })

            enc_sizes.append(len(enc_input_ids))
            dec_sizes.append(len(target) - 1)
            self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len
