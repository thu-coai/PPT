import torch

# path = "/mnt/sfs_turbo/back-folder/CPM-Finetune-gyx/results/poem_mc/full_data/t5_finetune_lr0.05const_G2_prompt_seed1234_save/acc_2000_0.932/2000/mp_rank_00_model_states.pt"

# ckpt = torch.load(path, map_location="cpu")

# t = ckpt["module"]["encoder.word_embeds.weight"].float()

# print(t.size())
# print(t[0])
# print(t.mean())
# t_n = torch.sqrt(torch.mean(t * t, dim=1))[:100]
# print(t_n)
# print(t_n.mean())

# for x in [300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 4500, 4800, 5100, 5400, 5700]:
#     print(x)
#     path = "/mnt/sfs_turbo/back-folder/CPM-Finetune-gyx/results/poem_mc/full_data/t5_finetune_lr0.05const_G2_prompt_seed1234_save/{}/{}/mp_rank_00_model_states.pt".format(x, x)
#     ckpt = torch.load(path, map_location="cpu")
#     t = ckpt["module"]["encoder.prompt_embeds.weight"].float()

#     print(t.size())
#     print(t[0])
#     print(t.mean())
#     t_n = torch.sqrt(torch.mean(t * t, dim=1))
#     print(t_n)
#     print(t_n.mean())

#     print()

for x in [100, 200, 300, 400, 500, 600, 700, 800]:
    print(x)
    path = "/mnt/sfs_turbo/back-folder/CPM-Finetune-gyx/results/poem_mc/lr0.05const_G1_prompt_bs4_num64_init_vocab_limit1000_fix_samp/seed20_save/{}/mp_rank_00_model_states.pt".format(x)

    ckpt = torch.load(path, map_location="cpu")

    t = ckpt["module"]["encoder.prompt_embeds.weight"].float()

    print(t.size())
    print(t[0])
    print(t[1])
    print(t.mean())
    t_n = torch.sqrt(torch.mean(t * t, dim=1))
    print(t_n)
    print(t_n.mean())
    print()

    # t = ckpt["module"]["encoder.word_embeds.weight"].float()

    # print(t.size())
    # print(t[0])
    # print(t.mean())
    # t_n = torch.sqrt(torch.mean(t * t, dim=1))[:100]
    # print(t_n)
    # print(t_n.mean())