import torch
import sys
import os

input_path = sys.argv[1]

ckpt = torch.load(os.path.join(input_path, "mp_rank_00_model_states.pt"), map_location="cpu")

t = ckpt["module"]["encoder.prompt_embeds.weight"]

torch.save(t, os.path.join(input_path, "prompt.pt"))