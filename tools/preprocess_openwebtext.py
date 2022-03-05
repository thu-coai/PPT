import re
import os
import sys
from multiprocessing import Pool

class P(object):
    def __init__(self, input_dir, output_dir, x):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.x = x

    def clean(self, text):
        p = r"\x00[^\x00]{1,30}\x00"

        m = re.search(p, text)
        while m is not None:
            s = m.span()
            text = text[:s[0]] + text[s[1]:]
            m = re.search(p, text)

        p1 = r"\00+"
        a = re.sub(p1, "", text)

        p2 = r"\d.{1,50}\.txt"
        doc_list = re.split(p2, a)

        new_doc_list = []
        for doc in doc_list:
            doc = doc.split("\n")
            doc = [x.strip() for x in doc if len(x.strip()) > 0]
            if len(doc) > 0:
                doc = "<@x(x!>".join(doc)
                new_doc_list.append(doc)

        new_doc_list = "\n".join(new_doc_list)

        return new_doc_list

    def process(self, i):
        file_path = os.path.join(self.input_dir, "urlsf_subset{}-{}_data".format(self.x, i))
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                text = f.read()
            text = self.clean(text)
            with open(os.path.join(self.output_dir, "{}_{}.txt".format(self.x, i)), "w") as f:
                f.write(text)

        return file_path


k = int(sys.argv[1])
input_dir = "/data/gyx/ppt_data/public/English/openwebtext/"
output_dir = "/data/gyx/ppt_data/public/English/pretrain_data_raw_3/"


for x in ["20"]:
    new_output_dir = os.path.join(output_dir, x)
    os.makedirs(new_output_dir, exist_ok=True)
    pro = P(input_dir, new_output_dir, x)
    pool = Pool(32)
    processed_doc = pool.imap_unordered(pro.process, range(0, 1000), 100)

    for file_path in processed_doc:
        pass

    pool.close()