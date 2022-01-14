import re
import os
import sys
from multiprocessing import Pool

class P(object):
    def __init__(self, input_dir, output_dir, x):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.x = x

    def process(self, i):
        file_path = os.path.join(self.input_dir, "{}_{}.txt".format(self.x, i))
        print(file_path)
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                lines = f.readlines()
            
            with open(os.path.join(self.output_dir, "{}_{}.txt".format(self.x, i)), "w") as f:
                for line in lines:
                    line = line.strip()
                    if len(line) > 10:
                        f.write(line + "\n")

        return file_path


k = int(sys.argv[1])
input_dir = "/mnt/sfs_turbo/gyx/data_en/pretrain_data_raw/"
output_dir = "/mnt/sfs_turbo/gyx/data_en/pretrain_data_raw_2/"

for x in ["20"]:
    new_output_dir = os.path.join(output_dir, x)
    os.makedirs(new_output_dir, exist_ok=True)
    new_input_dir = os.path.join(input_dir, x)
    pro = P(new_input_dir, new_output_dir, x)
    pool = Pool(32)
    processed_doc = pool.imap_unordered(pro.process, range(0, 1001), 100)

    for file_path in processed_doc:
        pass

    pool.close()
