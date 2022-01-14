import sys
import os
import torch

ckpt_dir = sys.argv[1]
save_path = sys.argv[2]

ckpt_path = os.path.join(ckpt_dir, "mp_rank_00_model_states.pt")

model = torch.load(ckpt_path)

prompt_embed = model["module"]["encoder.prompt_embeds.weight"]

torch.save(prompt_embed, save_path)