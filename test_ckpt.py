import sys
import torch


for i in range(4):
    ckpt1_path="/mnt/sfs_turbo/gyx/CPM-2-Pretrain-En/results/nsp_10g_3c_en_lr0.1/8000/8000/mp_rank_0{}_model_states.pt".format(i)
    ckpt2_path="/mnt/sfs_turbo/gyx/checkpoints/t5-xxl/t5-MP4/1/mp_rank_0{}_model_states.pt".format(i)

    model1 = torch.load(ckpt1_path)
    model2 = torch.load(ckpt2_path)

    model2["module"] = model1["module"]

    # torch.save(d, "/mnt/sfs_turbo/gyx/checkpoints/test_9_29/rng/rng_rank_0{}.pt".format(i))

    torch.save(model2, "/mnt/sfs_turbo/gyx/checkpoints/test_9_29_2/1/mp_rank_0{}_model_states.pt".format(i))