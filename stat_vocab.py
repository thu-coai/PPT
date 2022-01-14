import json
import random
from tokenization_t5 import EncDecTokenizer

# all_token_ids = {}

# for split in ["train", "valid"]:
#     with open("/mnt/sfs_turbo/gyx/data_en/sst2/{}.jsonl".format(split)) as f:
#         lines = f.readlines()

#     tokenizer = EncDecTokenizer("/mnt/sfs_turbo/gyx/PPT/sp_t5/spiece.model")

#     for line in lines:
#         line = json.loads(line)
#         token_ids = tokenizer.encode(line["sentence"])
#         for token_id in token_ids:
#             if token_id in all_token_ids:
#                 all_token_ids[token_id] += 1
#             else:
#                 all_token_ids[token_id] = 1

# x = sorted(all_token_ids.items(), key=lambda x: x[1], reverse=True)

# print(x)

# label_ids = [xx[0] for xx in x][:100]

# random.shuffle(label_ids)

# print(label_ids)

all_token_ids = {}

for split in ["train", "valid"]:
    with open("/mnt/sfs_turbo/gyx/data_en/boolq/{}.jsonl".format(split)) as f:
        lines = f.readlines()

    tokenizer = EncDecTokenizer("/mnt/sfs_turbo/gyx/PPT/sp_t5/spiece.model")

    for line in lines:
        line = json.loads(line)
        for key_name in ["question", "passage"]:
            token_ids = tokenizer.encode(line[key_name])
            for token_id in token_ids:
                if token_id in all_token_ids:
                    all_token_ids[token_id] += 1
                else:
                    all_token_ids[token_id] = 1

x = sorted(all_token_ids.items(), key=lambda x: x[1], reverse=True)

print(x)

label_ids = [xx[0] for xx in x][:100]

random.shuffle(label_ids)

print(label_ids)