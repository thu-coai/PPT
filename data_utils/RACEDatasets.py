import json

from tokenization_t5 import EncDecTokenizer
from .EncDecDatasets import EncDecDataset

class RACEDataset(EncDecDataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(RACEDataset, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)
    
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
                lim = 612 - len(question_ids) - len(choice_ids) - prompt_len - 10
                context_ids = context_ids[:lim]
                enc_input_ids = [-(i + 1) for i in range(prompt_len)] + context_ids + [117] + question_ids + [58] + choice_ids + [58] + self.tokenizer.encode("The correct one is ") + [self.tokenizer.get_sentinel_id(0)]
            else:
                lim = 512 - len(question_ids) - len(choice_ids) - 10
                context_ids = context_ids[:lim]
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


class RACEDatasetUni(EncDecDataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(RACEDatasetUni, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)
    
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
                lim = 612 - len(question_ids) - len(choice_ids) - prompt_len - 10
                context_ids = context_ids[:lim]
                enc_input_ids = [-(i + 1) for i in range(prompt_len)] + context_ids + [117] + question_ids + [58] + choice_ids + [58] + self.tokenizer.encode("The correct one is ") + [self.tokenizer.get_sentinel_id(0)]
            else:
                lim = 512 - len(question_ids) - len(choice_ids) - 10
                context_ids = context_ids[:lim]
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
