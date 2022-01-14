import sys
import os
import torch

ckpt_dir = sys.argv[1]
save_path = sys.argv[2]

ckpt_path = os.path.join(ckpt_dir, "mp_rank_00_model_states.pt")

model = torch.load(ckpt_path)

prompt_embed = model["module"]["encoder.prompt_embeds.weight"]

prompt_embed[5][10] = 0.05
prompt_embed[6][11] = 0.03
prompt_embed[7][12] = 0.02
prompt_embed[8][13] = 0.012
prompt_embed[9][14] = -0.05
prompt_embed[10][15] = -0.032
prompt_embed[11][16] = -0.15
prompt_embed[12][17] = -0.274


torch.save(prompt_embed, save_path)