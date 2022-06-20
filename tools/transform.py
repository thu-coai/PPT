#coding:utf-8
import torch
import argparse
import os
import tqdm
import copy


def transform_new_model(model_hf):
    model_new = {}

    cnt = 0

    for i in range(24):
        # encoder
        target_k = 'encoder.blocks.{}.self_attn.self_attn.project.weight'.format(i)
        source = ['encoder.block.{}.layer.0.SelfAttention.q.weight'.format(i), 'encoder.block.{}.layer.0.SelfAttention.k.weight'.format(i), 'encoder.block.{}.layer.0.SelfAttention.v.weight'.format(i)]
        # qkv
        model_new[target_k] = torch.cat([model_hf[x] for x in source], 0)
        cnt += 3

        target_k = 'encoder.blocks.{}.self_attn.self_attn.dense.weight'.format(i)
        source = 'encoder.block.{}.layer.0.SelfAttention.o.weight'.format(i)
        model_new[target_k] = model_hf[source] / 100
        cnt += 1

        target_k = 'encoder.blocks.{}.self_attn.layer_norm.weight'.format(i)
        source = 'encoder.block.{}.layer.0.layer_norm.weight'.format(i)
        model_new[target_k] = model_hf[source]
        cnt += 1

        target_k = 'encoder.blocks.{}.ff.dense_relu_dense.wi_0.weight'.format(i)
        source = 'encoder.block.{}.layer.1.DenseReluDense.wi_0.weight'.format(i)
        model_new[target_k] = model_hf[source]
        cnt += 1

        target_k = 'encoder.blocks.{}.ff.dense_relu_dense.wi_1.weight'.format(i)
        source = 'encoder.block.{}.layer.1.DenseReluDense.wi_1.weight'.format(i)
        model_new[target_k] = model_hf[source] / 10
        cnt += 1

        target_k = 'encoder.blocks.{}.ff.dense_relu_dense.wo.weight'.format(i)
        source = 'encoder.block.{}.layer.1.DenseReluDense.wo.weight'.format(i)
        model_new[target_k] = model_hf[source] / 10
        cnt += 1

        target_k = 'encoder.blocks.{}.ff.layer_norm.weight'.format(i)
        source = 'encoder.block.{}.layer.1.layer_norm.weight'.format(i)
        model_new[target_k] = model_hf[source]
        cnt += 1

        # decoder
        target_k = 'decoder.blocks.{}.self_attn.self_attn.project.weight'.format(i)
        source = ['decoder.block.{}.layer.0.SelfAttention.q.weight'.format(i), 'decoder.block.{}.layer.0.SelfAttention.k.weight'.format(i), 'decoder.block.{}.layer.0.SelfAttention.v.weight'.format(i)]
        # qkv
        model_new[target_k] = torch.cat([model_hf[x] for x in source], 0)
        cnt += 3

        target_k = 'decoder.blocks.{}.cross_attn.cross_attn.project_kv.weight'.format(i)
        source = ['decoder.block.{}.layer.1.EncDecAttention.k.weight'.format(i), 'decoder.block.{}.layer.1.EncDecAttention.v.weight'.format(i)]
        # kv
        model_new[target_k] = torch.cat([model_hf[x] for x in source], 0)
        cnt += 2

        target_k = 'decoder.blocks.{}.cross_attn.cross_attn.project_q.weight'.format(i)
        source = 'decoder.block.{}.layer.1.EncDecAttention.q.weight'.format(i)
        model_new[target_k] = model_hf[source]
        cnt += 1

        target_k = 'decoder.blocks.{}.cross_attn.cross_attn.dense.weight'.format(i)
        source = 'decoder.block.{}.layer.1.EncDecAttention.o.weight'.format(i)
        model_new[target_k] = model_hf[source] / 100
        cnt += 1

        target_k = 'decoder.blocks.{}.cross_attn.layer_norm.weight'.format(i)
        source = 'decoder.block.{}.layer.1.layer_norm.weight'.format(i)
        model_new[target_k] = model_hf[source]
        cnt += 1

        target_k = 'decoder.blocks.{}.self_attn.self_attn.dense.weight'.format(i)
        source = 'decoder.block.{}.layer.0.SelfAttention.o.weight'.format(i)
        model_new[target_k] = model_hf[source] / 100
        cnt += 1

        target_k = 'decoder.blocks.{}.self_attn.layer_norm.weight'.format(i)
        source = 'decoder.block.{}.layer.0.layer_norm.weight'.format(i)
        model_new[target_k] = model_hf[source]
        cnt += 1

        target_k = 'decoder.blocks.{}.ff.dense_relu_dense.wi_0.weight'.format(i)
        source = 'decoder.block.{}.layer.2.DenseReluDense.wi_0.weight'.format(i)
        model_new[target_k] = model_hf[source]
        cnt += 1

        target_k = 'decoder.blocks.{}.ff.dense_relu_dense.wi_1.weight'.format(i)
        source = 'decoder.block.{}.layer.2.DenseReluDense.wi_1.weight'.format(i)
        model_new[target_k] = model_hf[source] / 10
        cnt += 1

        target_k = 'decoder.blocks.{}.ff.dense_relu_dense.wo.weight'.format(i)
        source = 'decoder.block.{}.layer.2.DenseReluDense.wo.weight'.format(i)
        model_new[target_k] = model_hf[source] / 10
        cnt += 1

        target_k = 'decoder.blocks.{}.ff.layer_norm.weight'.format(i)
        source = 'decoder.block.{}.layer.2.layer_norm.weight'.format(i)
        model_new[target_k] = model_hf[source]
        cnt += 1

    source = 'shared.weight'
    target_k = 'word_embeds.weight'

    embeds = model_hf[source]

    model_new[target_k] = embeds / 100
    target_k = 'encoder.word_embeds.weight'
    model_new[target_k] = embeds / 100
    target_k = 'decoder.word_embeds.weight'
    model_new[target_k] = embeds / 100
    cnt += 3

    source = 'encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight'
    target_k = 'encoder.blocks.0.self_attn.self_attn.relative_attention_bias.weight'
    model_new[target_k] = model_hf[source]
    cnt += 1

    source = 'decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight'
    target_k = 'decoder.blocks.0.self_attn.self_attn.relative_attention_bias.weight'
    model_new[target_k] = model_hf[source]
    cnt += 1

    source = 'lm_head.weight'
    target_k = 'lm_head.weight'

    embeds = model_hf[source]

    model_new[target_k] = embeds
    cnt += 1

    source = 'encoder.final_layer_norm.weight'
    target_k = 'encoder.final_layernorm.weight'
    model_new[target_k] = model_hf[source]
    cnt += 1

    source = 'decoder.final_layer_norm.weight'
    target_k = 'decoder.final_layernorm.weight'
    model_new[target_k] = model_hf[source]
    cnt += 1

    print("new module number:", cnt, "origin module number:", len(model_hf))

    return {'module': model_new}


def change_mp(d, output_dir, mp_size, half=False):

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
    ]

    dd = {}
    dd['lr_scheduler'] = {}

    dd['lr_scheduler']['num_iters'] = 1
    dd['lr_scheduler']['start_lr'] = 0.001
    dd['lr_scheduler']['warmup_iter'] = 10000

    dd['skipped_steps'] = 0
    dd['global_steps'] = 1
    dd['global_samples'] = 100
    dd['iteration'] = 1
    dd['dp_world_size'] = 1

    print("Increase MP size.")
    ratio = mp_size

    start = 0
    end = ratio

    for j in tqdm.tqdm(range(start, end)):
        d_new = {}
        shift = j - start
        for k, v in dd.items():
            if k != 'module':
                if k in preserve_keys:
                    d_new[k] = copy.deepcopy(dd[k])
                elif k == "mp_world_size":
                    d_new[k] = ratio
                else:
                    d_new[k] = None
        d_new['module'] = {}
        for k, v in d['module'].items():
            assert len(v.shape) < 3
            if len(v.shape) == 2:
                if 'project.weight' in k:
                    part = v.shape[0] // ratio // 3
                    d_new['module'][k] = torch.cat([v[shift*part:(shift+1)*part, :], v[(shift+ratio)*part:(
                        shift+1+ratio)*part, :], v[(shift+2*ratio)*part:(shift+1+2*ratio)*part, :]], 0)
                elif 'project_q.weight' in k:
                    part = v.shape[0] // ratio
                    d_new['module'][k] = v[shift*part:(shift+1)*part, :]
                elif 'project_kv.weight' in k:
                    part = v.shape[0] // ratio // 2
                    d_new['module'][k] = torch.cat(
                        [v[shift*part:(shift+1)*part, :], v[(shift+ratio)*part:(shift+1+ratio)*part, :]], 0)
                elif 'word_embeds.weight' in k or 'dense_relu_dense.wi_1.weight' in k or 'dense_relu_dense.wi_0.weight' in k or 'lm_head.weight' in k:
                    part = v.shape[0] // ratio
                    d_new['module'][k] = v[shift*part:(shift+1)*part, :]
                else:
                    part = v.shape[1] // ratio
                    d_new['module'][k] = v[:, shift*part:(shift+1)*part]
            else:
                d_new['module'][k] = v

            if half:
                d_new['module'][k] = d_new['module'][k].half()

        filename = os.path.join(output_dir, "1", "mp_rank_0{}_model_states.pt".format(j))
        torch.save(d_new, filename)


def main():
    parser = argparse.ArgumentParser("Transform huggingface checkpoints to megatron+deepspeed checkpoints")

    parser.add_argument("--hf_path", type=str)
    parser.add_argument("--mp_size", type=int, default=4)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--half", action="store_true")

    args = parser.parse_args()

    model_hf = torch.load(args.hf_path, map_location='cpu')

    new_model = transform_new_model(model_hf)

    change_mp(new_model, args.save_path, args.mp_size, half=args.half)
    
if __name__ == '__main__':
    main()
