import torch

model1 = torch.load("/mnt/sfs_turbo/gyx/PPT/results/sst2/full-repro/at_large_fp32_fix_only_at/lr0.00005_G1/seed1234/30/mp_rank_00_model_states.pt", map_location="cpu")
model2 = torch.load("/mnt/sfs_turbo/gyx/PPT/results/sst2/full-repro/at_large_fp32_fix_only_at/lr0.00005_G1/seed1234/60/mp_rank_00_model_states.pt", map_location="cpu")


print(model1["module"]["decoder.blocks.18.ff.adapter.adapter_up.weight"])
print(model2["module"]["decoder.blocks.18.ff.adapter.adapter_up.weight"])

print(model1["module"]["decoder.blocks.11.ff.dense_relu_dense.wo.weight"])
print(model2["module"]["decoder.blocks.11.ff.dense_relu_dense.wo.weight"])
