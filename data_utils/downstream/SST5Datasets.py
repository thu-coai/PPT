import json
from tokenization_t5 import EncDecTokenizer
from .EncDecDatasets import EncDecDataset


class SST5Dataset(EncDecDataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(SST5Dataset, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):

        self.label_map = {
            "0": "terrible",
            "1": "bad",
            "2": "maybe",
            "3": "good",
            "4": "great"
        }

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

            target = [0, self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode(self.label_map[d["label"]])

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


class SST5DatasetUni(EncDecDataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(SST5DatasetUni, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):

        self.label_map = {
            "0": 188,
            "1": 279,
            "2": 254,
            "3": 308,
            "4": 427
        }

        data = []
        enc_sizes = []
        dec_sizes = []

        with open(self.path, "r") as f:
            lines = f.readlines()

        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)
            sid = self.tokenizer.encode(d["sentence"])

            if self.prompt_config:
                choice_ids = [188, 5] + self.tokenizer.encode("terrible") + [5] + \
                             [279, 5] + self.tokenizer.encode("bad") + [5] + \
                             [254, 5] + self.tokenizer.encode("maybe") + [5] + \
                             [308, 5] + self.tokenizer.encode("good") + [5] + \
                             [427, 5] + self.tokenizer.encode("great") + [5]

                prompt_len = self.prompt_config["enc"]["prompt_len"]
                context = [-(i + 1) for i in range(prompt_len)] + sid + [58] + choice_ids + [58] + self.tokenizer.encode("The correct one is ") + [self.tokenizer.get_sentinel_id(0)]
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
