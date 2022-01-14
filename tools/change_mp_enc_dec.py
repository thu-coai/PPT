import sys
import os
import torch
import copy
import tqdm

# filenames = ["/mnt/sfs_turbo/enc-dec-pretrain/results/enc_dec_mt5_2021-4-19/124000/merge.pt"]

# filenames = ["/mnt/sfs_turbo/mt5-origin.bin"]
# filenames = ['/mnt/sfs_turbo/mt5_origin/1/mp_rank_00_model_states.pt', '/mnt/sfs_turbo/mt5_origin/1/mp_rank_01_model_states.pt', '/mnt/sfs_turbo/mt5_origin/1/mp_rank_02_model_states.pt', '/mnt/sfs_turbo/mt5_origin/1/mp_rank_03_model_states.pt']
filenames = ["/mnt/sfs_turbo/gyx/checkpoints/t5-xxl-lm/new_model.bin"]

# filenames = ["/mnt/sfs_turbo/enc-dec-pretrain/results/enc_dec_mt5_2021-4-19/108000/mp_rank_0{}_model_states.pt".format(i) for i in range(4)]

# output_dir = "/mnt/sfs_turbo/mt5_lr_0.00001_wm_1/"

# output_dir = "/mnt/sfs_turbo/mt5_108000"

output_dir = "/mnt/sfs_turbo/gyx/checkpoints/t5-xxl-lm/t5-MP4/"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "1"), exist_ok=True)

with open(os.path.join(output_dir, "latest_checkpointed_iteration.txt"), "w") as f:
    f.write(str(1) + "\n")

preserve_keys = [
    "lr_scheduler",
    "skipped_steps",
    "global_steps",
    "global_samples",
    "dp_world_size",
    "iteration",
    "np_rng_state",
    "random_rng_state",
    "torch_rng_state",
    "cuda_rng_state",
    "rng_tracker_states",
    
]

dd = torch.load('/mnt/sfs_turbo/gyx/checkpoints/enc_dec_mt5_2021-4-15/0/mp_rank_00_model_states.pt', map_location='cpu')

dd['lr_scheduler']['num_iters'] = 1
dd['lr_scheduler']['start_lr'] = 0.001
dd['lr_scheduler']['warmup_iter'] = 10000

dd['skipped_steps'] = 0
dd['global_steps'] = 1
dd['global_samples'] = 100
dd['iteration'] = 1

print("Increase MP size.")
ratio = 4
for i in range(len(filenames)):
    start = ratio * i
    end = ratio * (i+1)
    d = torch.load(filenames[i], map_location='cpu')
    for j in tqdm.tqdm(range(start, end)):
        d_new = {}
        shift = j - start
        for k, v in dd.items():
            if k != 'module':
                if k in preserve_keys:
                    d_new[k] = copy.deepcopy(dd[k])
                elif k == "mp_world_size":
                    d_new[k] = ratio * len(filenames)
                else:
                    d_new[k] = None
        d_new['module'] = {}
        for k, v in d['module'].items():
            assert len(v.shape) < 3
            if len(v.shape) == 2:
                if 'project.weight' in k:
                    part = v.shape[0] // ratio // 3
                    d_new['module'][k] = torch.cat([v[shift*part:(shift+1)*part, :], v[(shift+ratio)*part:(shift+1+ratio)*part, :], v[(shift+2*ratio)*part:(shift+1+2*ratio)*part, :]], 0)
                elif 'project_q.weight' in k:
                    part = v.shape[0] // ratio
                    d_new['module'][k] = v[shift*part:(shift+1)*part, :]
                elif 'project_kv.weight' in k:
                    part = v.shape[0] // ratio // 2
                    d_new['module'][k] = torch.cat([v[shift*part:(shift+1)*part, :], v[(shift+ratio)*part:(shift+1+ratio)*part, :]], 0)
                elif 'word_embeds.weight' in k or 'dense_relu_dense.wi_1.weight' in k or 'dense_relu_dense.wi_0.weight' in k or 'lm_head.weight' in k:
                    part = v.shape[0] // ratio
                    d_new['module'][k] = v[shift*part:(shift+1)*part, :]
                else:
                    part = v.shape[1] // ratio
                    d_new['module'][k] = v[:, shift*part:(shift+1)*part]
            else:
                d_new['module'][k] = v

            # d_new['module'][k] = d_new['module'][k].half()

            
        filename = os.path.join(output_dir, "1", "mp_rank_0{}_model_states.pt".format(j))
        torch.save(d_new, filename)
