import sys
import torch

prompt_embed = torch.load("/mnt/sfs_turbo/gyx/PPT/prompt_embeds/pretrain-nsp_10g_3c_en_lr0.1-8000")

for i in range(4):
    ckpt1_path="/mnt/sfs_turbo/gyx/checkpoints/t5-xxl/t5-MP4/1/mp_rank_0{}_model_states.pt".format(i)

    model1 = torch.load(ckpt1_path)

    model1["module"]["encoder.prompt_embeds.weight"] = prompt_embed

    print(prompt_embed)

    # torch.save(d, "/mnt/sfs_turbo/gyx/checkpoints/test_9_29/rng/rng_rank_0{}.pt".format(i))

    torch.save(model1, "/mnt/sfs_turbo/gyx/checkpoints/test_9_29_4/1/mp_rank_0{}_model_states.pt".format(i))