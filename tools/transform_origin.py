#coding:utf-8
import torch

model_hf = torch.load("/mnt/sfs_turbo/gyx/checkpoints/t5-xxl-lm/pytorch_model.bin", map_location='cpu')

print(model_hf.keys())

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

# mt5_vocab = open('/mnt/sfs_turbo/mt5_vocab.txt')
# mt5_vocab.readline()
# mt5 = [(line.strip(), i) for i, line in enumerate(mt5_vocab.readlines())]
# mt5 = dict(mt5)

# vec = [line.strip().replace('：', ":").replace('，', ",").replace('。', ".").replace('？', "?").replace('！', "!").replace('（', "(").replace('）', ")") for line in open('/mnt/sfs_turbo/enc-dec-pretrain/bpe_new/vocab.txt').readlines()]

source = 'shared.weight'
target_k = 'word_embeds.weight'

import random

new_tensor = []
embeds = model_hf[source]
# for i in range(26050):
#     if vec[i] in mt5:
#         new_tensor.append(embeds[mt5[vec[i]], :])
#     elif vec[i][:1] in mt5:
#         new_tensor.append(embeds[mt5[vec[i][:1]], :])
#     else:
#         # new_tensor.append(torch.nn.init.normal_(torch.ones_like(embeds[0, :]), mean=0.0, std=0.001))
#         new_tensor.append(embeds[random.choice(list(range(250000))), :])
# new_tensor[1] = embeds[0, :]
# new_tensor = torch.stack(new_tensor, 0)

# extract_ids = model_hf[source][250000:250100, :].flip(0)
# new_tensor = torch.cat([new_tensor, extract_ids, extract_ids[10:100, :]], 0)

# new_tensor = torch.cat([model_hf[source][:26050, :], model_hf[source][250000:250100, :], model_hf[source][250010:250100, :]], 0)
# new_tensor = model_hf[source][:26240, :]
model_new[target_k] = embeds / 100
target_k = 'encoder.word_embeds.weight'
model_new[target_k] = embeds / 100
target_k = 'decoder.word_embeds.weight'
model_new[target_k] = embeds / 100
cnt += 3

# target_k = 'encoder.position_embeds.weight'
# new_tensor = torch.zeros([512, 4096])
# new_tensor = torch.nn.init.normal_(new_tensor, mean=0.0, std=0.0001)
# model_new[target_k] = new_tensor

# target_k = 'decoder.position_embeds.weight'
# new_tensor = torch.zeros([512, 4096])
# new_tensor = torch.nn.init.normal_(new_tensor, mean=0.0, std=0.0001)
# model_new[target_k] = new_tensor

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

new_tensor = []
embeds = model_hf[source]
# for i in range(26050):
#     if vec[i] in mt5:
#         new_tensor.append(embeds[mt5[vec[i]], :])
#     elif vec[i][:1] in mt5:
#         new_tensor.append(embeds[mt5[vec[i][:1]], :])
#     else:
#         # new_tensor.append(torch.nn.init.normal_(torch.ones_like(embeds[0, :]), mean=0.0, std=0.001))
#         new_tensor.append(embeds[random.choice(list(range(250000))), :])
# new_tensor[1] = embeds[0, :]
# new_tensor = torch.stack(new_tensor, 0)

# extract_ids = model_hf[source][250000:250100, :].flip(0)

# new_tensor = torch.cat([new_tensor, extract_ids, extract_ids[10:100, :]], 0)

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

# 'encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight'
# 'decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight'
# 'decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight'

print(cnt, len(model_hf))

torch.save({'module': model_new}, "/mnt/sfs_turbo/gyx/checkpoints/t5-xxl-lm/new_model.bin")