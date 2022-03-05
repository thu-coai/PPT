import json
from tokenization_t5 import EncDecTokenizer
from .EncDecDatasets import EncDecDataset


class BoolQDataset(EncDecDataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(BoolQDataset, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):

        data = []
        enc_sizes = []
        dec_sizes = []

        with open(self.path, "r") as f:
            lines = f.readlines()

        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)

            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                qid = self.tokenizer.encode(d["question"])
                max_passage_len = 512 - prompt_len - 1 - len(qid)
                pid = self.tokenizer.encode(d["passage"])[:max_passage_len]
                context = [-(i + 1) for i in range(prompt_len)] + qid + [self.tokenizer.get_sentinel_id(0)] + pid
            else:
                qid = self.tokenizer.encode(d["question"])
                max_passage_len = 512 - len(qid) - len(self.tokenizer.encode("question: ")) - len(self.tokenizer.encode(" passage: "))
                pid = self.tokenizer.encode(d["passage"])[:max_passage_len]
                context = self.tokenizer.encode("question: ") + qid + [58] + [self.tokenizer.get_sentinel_id(0)] + [5] + pid

            target = [0, self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode("yes" if d["label"] else "no")

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


class BoolQDatasetUni(EncDecDataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(BoolQDatasetUni, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):

        data = []
        enc_sizes = []
        dec_sizes = []

        choice_ids = [188, 5] + self.tokenizer.encode("no") + [5] + [279, 5] + self.tokenizer.encode("yes") + [5]

        with open(self.path, "r") as f:
            lines = f.readlines()

        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)

            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                qid = self.tokenizer.encode(d["question"])
                max_passage_len = 512 - prompt_len - 1 - len(qid)
                pid = self.tokenizer.encode(d["passage"])[:max_passage_len]
                context = [-(i + 1) for i in range(prompt_len)] + qid + [58] + pid + [5] + choice_ids + [58] + self.tokenizer.encode("The correct one is ") + [self.tokenizer.get_sentinel_id(0)]
            else:
                pass

            target = [0, self.tokenizer.get_sentinel_id(0)] + [279 if d["label"] else 188]

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
