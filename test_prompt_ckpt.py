import torch
import os

# input_prompt_dir = "/mnt/sfs_turbo/gyx/PPT/results/qa/race_string/t5_xxl_lr0.001_G8_prompt_save/seed10/"
input_prompt_dir = "/mnt/sfs_turbo/gyx/PPT/results/qa/race_string/lr0.001_G8_prompt_fp16_save_mp4/seed10/"

for i in [100, 200, 300]:
    prompt = torch.load(os.path.join(input_prompt_dir, "prompt-{}.pt".format(i)), map_location="cpu")

    print(i)
    print(len(prompt[0]))