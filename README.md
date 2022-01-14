# CPM-Finetune

本仓库为CPM模型的 fine-tune 代码仓库，可以用于模型 fine-tune 的多机多卡训练/测试。目前支持了 ChID 中文成语填空数据集和 STC 中文对话数据集。[[项目首页](https://cpm.baai.ac.cn)] [[模型下载](https://cpm.baai.ac.cn/download.html)] [[技术报告](https://arxiv.org/abs/2012.00413)]

同时，该仓库也提供了 ChID 数据集 zero-shot setting 下测试代码。

ChID 数据集来源于论文 [ChID: A Large-scale Chinese IDiom Dataset for Cloze Test](https://www.aclweb.org/anthology/P19-1075/). 

STC 数据集来源于论文 [Neural Responding Machine for Short-Text Conversation](https://www.aclweb.org/anthology/P15-1152/).

两个数据集可从[这里](https://drive.google.com/drive/folders/1gL01xbFBcrgP0TmgOhJ_uplkeG-BCwvM)下载。


## 1 安装

首先安装 pytorch 等基础依赖，再安装[APEX](https://github.com/NVIDIA/apex#quick-start)以支持 fp16，然后安装 deepspeed：

**安装基础依赖：**

```[bash]
pip install -r requirements.txt
```

**安装 apex：**

```[bash]
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

考虑apex的安装容易发生问题，我们构建了对应的Docker容器，可以进行快速环境搭建。安装方式如下：

```[bash]
docker pull dmye/cpm:v0
```

参考运行指令如下：

```[bash]
sudo docker run --gpus '"device=0,1"' -it -v <path>:/CPM  --name=cpm  cpm:v0
```

其中`<path>`为代码所在目录，-v进行文件目录挂载

**安装 deepspeed**

我们使用了以下版本的 deepspeed：
<https://github.com/microsoft/DeepSpeed/tree/f0f2a7026877d3fd2f6f5565a67cdffc89750cf0>
可根据其提供的文档进行安装。


## 2 Fine-Tune

### 2.1 数据预处理

### 2.1.1 ChiD
```[bash]
python3 preprocess_chid_finetune.py --data_dir ${PATH_TO_DATA_DIR} --tokenizer_path ${PATH_TO_TOKENIZER} --output_dir ${PATH_TO_OUTPUT}
```

其中，模板定义与实现在 preprocess_chid_finetune.py 文件 `process_one_sent` 函数中。最终，该文件生成的数据格式为：

```[python]
[
    {
        "sent": [8, 15, ....], # 经过 bpe 分词之后 token 对应的 id
        "truth": 3 # 正确答案成语的编号（0~9之间的整数）
    }
    ...
]
```

预处理完成后，指定的输出目录下会生成 train.json, valid.json, test.json 三个文件。

### 2.1.2 STC

```[bash]
python3 preprocess_stc_finetune.py --data_dir ${PATH_TO_DATA_DIR} --output_dir ${PATH_TO_OUTPUT}
```

该文件会将每段对话写成一行，上文前加入“对话上文:”，下文前加入“回复:”：
```
对话上文:二十六年前的我挺瘦吧？不知这几位盲童现在好吗？ 回复:恩，不但瘦，头发还很多。
```

注意：由于 STC 数据集很大，我们在数据预处理的时候切了其训练集的前 10% 以方便使用者测试 Fine-tune。

### 2.2 Fine-Tune 训练/测试

进行 fine-tune 训练的时候可以选择 fp16 或者 fp32。我们在实验中发现，在使用 fp16 进行训练的时候，需要加载预训练时的动量才能使模型较快收敛，而采用 fp32 训练则不会有这个问题。因此，我们推荐直接使用 fp32 进行 fine-tune 训练。

另外，关于内存使用，我们使用了 deepspeed 的 activation checkpointing 以节省内存，相关选项已经在运行脚本中修改。

#### ChID:

```[bash]
bash scripts/chid/finetune_chid_large.sh # for fp16 fine-tune
bash scripts/chid/finetune_chid_large_fp32.sh # for fp32 fine-tune
bash scripts/chid/finetune_chid_large_fp32_multinode.sh # for multi-node fp32 fine-tune
```

#### STC:
```[bash]
bash scripts/language_model/finetune_lm_large.sh # for fp16 fine-tune
bash scripts/language_model/finetune_lm_large_fp32.sh # for fp32 fine-tune
bash scripts/language_model/finetune_lm_large_fp32_multinode.sh # for multi-node fp32 fine-tune
```

运行脚本之前，需要先将脚本中以下变量更改为实际的路径：

```[bash]
DATA_DIR # 预处理后数据的目录
CHECKPOINT_PATH # 预训练结束后模型的路径
RESULTS_DIR # 训练结果的存放处
MODEL_NAME # 给模型起的名字
TOKENIZER_PATH # tokenizer 的路径
```

如果要进行多机训练，可能还需要修改
```[bash]
NUM_WORKERS # 节点数量
NUM_GPUS_PER_WORKER # 每个节点的卡数
```
以及 `scripts/host_files/hostfile` 文件。具体格式可以参考 deepspeed 的[官方文档](https://www.deepspeed.ai/getting-started/)

进行测试之前，需要去掉脚本中 `--do-train` 选项，然后可以使用 `--eval_ckpt_path` 选项来指定需要测试的模型。


## 3 Zero-Shot

### 3.1 数据预处理

```[bash]
python3 preprocess_chid_zeroshot.py --data_dir ${PATH_TO_DATA_DIR} --tokenizer_path ${PATH_TO_TOKENIZER} --output_dir ${PATH_TO_OUTPUT}
```

该文件会将每个候选的成语填入文章相应的空白中，每个空白生成10个新的候选文章。最终，该文件生成的数据格式为：

```[python]

{
    "contents": [
        [8, 15, ....],
        ....
    ], # 所有样本经过 bpe 分词之后 token 对应的 id。
    "sids": [
        0,
        0,
        ...
        1,
        1,
        ...
    ], # 每个生成出的候选文章对应原来样本的编号
    "cids": [
        0,
        1,
        2,
        ...
        9,
        0,
        1,
        ...
    ], # 每个生成出的候选文章对应的成语的编号
    "labels": [
        3,
        2,
        ...
    ], # 每个原样本的正确答案编号（0~9之间的整数）
}
```

与处理完成后，指定的输出目录下会生成 test.json 文件。

### 3.2 Zero-Shot 测试

```[bash]
bash scripts/chid/zero-shot_chid_large.sh
```

运行脚本之前，需要先将脚本中以下变量更改为实际的路径：

```[bash]
DATA_DIR # 预处理后数据的目录
CHECKPOINT_PATH # 预训练结束后模型的路径
RESULTS_DIR # 训练结果的存放处
MODEL_NAME # 给模型起的名字
TOKENIZER_PATH # tokenizer 的路径
```

## 4 参考性能

|            | Fine-Tune | Zero-Shot |
| ---------- | --------- | --------- |
| CPM-small  | 0.657     | 0.433     |
| CPM-medium | 0.695     | 0.524     |
| CPM-large  | **0.804** | **0.685** |

## 5 引用

```[latex]
@article{cpm-v1,
  title={CPM: A Large-scale Generative Chinese Pre-trained Language Model},
  author={Zhang, Zhengyan and Han, Xu, and Zhou, Hao, and Ke, Pei, and Gu, Yuxian and Ye, Deming and Qin, Yujia and Su, Yusheng and Ji, Haozhe and Guan, Jian and Qi, Fanchao and Wang, Xiaozhi and Zheng, Yanan and Zeng, Guoyang and Cao, Huanqi and Chen, Shengqi and Li, Daixuan and Sun, Zhenbo and Liu, Zhiyuan and Huang, Minlie and Han, Wentao and Tang, Jie and Li, Juanzi and Sun, Maosong},
  year={2020}
}
```
