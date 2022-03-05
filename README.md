# PPT

Code and datasets for our paper "PPT: Pre-trained Prompt Tuning for Few-shot Learning"



## 1 Environment

The code requires the CUDA10.2 toolkit. 

##### Install basic dependencies

```bash
pip install -r requirements.txt
```

##### Install apex

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
##### Install DeepSpeed

The version we used is `v0.3.9`, It can be installed from its [repo](https://github.com/microsoft/DeepSpeed/releases/tag/v0.3.9) or 
```bash
pip install deepspeed==0.3.9
```
Since there exist some **bugs** in DeepSpeed, you need to make some little modifications to this package. You can refer to this [issue](https://github.com/TsinghuaAI/CPM-2-Finetune/issues/11) for more information. Specifically, you need to modify two lines of code in `${PATH_TO_PYTHON_SITE_PACKAGE}/deepspeed/runtime/zero/stage1.py` and `${PATH_TO_PYTHON_SITE_PACKAGE}/deepspeed/runtime/engine.py`. We provide the modified `src/ds_fix/stage1.py` and `src/ds_fix/engine.py` in our repo. You can simply replace `${PATH_TO_PYTHON_SITE_PACKAGE}/deepspeed/runtime/zero/stage1.py` with `stage1.py` and `${PATH_TO_PYTHON_SITE_PACKAGE}/deepspeed/runtime/engine.py` with `engine.py` that we provided. 



## 2 Datasets

## 2.1 Downstream Datasets

The original datasets is obtained from [huggingface](https://huggingface.co/datasets).

The preprocessed datasets can be obtained from this link. If you do tuning (FT, PT, or PPT), you need to put the preprocessed data in `downstream_data/`.

## 2.2 Pre-training Data

Our pre-training data is sampled from [openwebtext](https://huggingface.co/datasets/openwebtext/tree/main). If you would like to preprocess the data from scratch, please put the `openwebtext.txt` in  `pretrain_data/raw/`. Run the following preprocessing scripts to construct the pre-training data:

```bash
bash scripts/tools/preprocess_pretrain_nsp.sh # Next Sentence Prediction
bash scripts/tools/preprocess_pretrain_nss.sh # Next Sentence Selection
bash scripts/tools/preprocess_pretrain_cls.sh # Single Sentence Classification
bash scripts/tools/preprocess_pretrain_nss_uni.sh # Unified Next Sentence Selection (for Unified PPT)
```

For reproductivity, we also provided the preprocessed pre-training data in this link. You can directly move the preprocessed pre-training data to `pretrain_data/preprocessed/`.



## 3 Pre-trained Checkpoints

## 3.1 Base Model

The original base model is obtained from [huggingface](https://huggingface.co/models). Before runing the code, please use the transforming scripts to transfer the original `pytorch_model.bin` model checkpoints to fit in our `deepspeed + megatron` framework:

```bash
mkdir -p checkpoints/t5-xxl/t5-MP4

python3 tools/transform.py \
--hf_path ${PATH_TO_PYTORCH_MODLE_BIN}
--save_path "./checkpoints/t5-xxl/t5-MP4"
```

## 3.2 Prompts

The pretrained prompts can be obtained from this link. You need to move the pre-tained prompts to `pretrained_prompts/`.



## 4 Run the code

All scripts are in the directory `scripts`.

Before running the code, please first change the `WORKING_DIR` to the current directory of this repo. If you are runing multiple scripts on a single node, you need to make sure that the `MASTER_PORT` of each script is different. 

If the checkpoint is successfully loaded, the log printed to the stdout should contain messages like `successfully loaded /path-to-checkpoint/t5-MP4/mp_rank_01_model_states.pt`. Otherwise, `WARNING: could not find the metadata file /***/latest_checkpointed_iteration.txt will not load any checkpoints and will start from random` will display. Note that when you successfully load the model, you will see messages like `The following zero checkpoints paths are missing: ['/path-to-checkpoint/eva/200000/zero_pp_rank_0_mp_rank_00_optim_states.pt',...` which mean optimizer states are not loaded. This **DOES NOT** affect the use of model inference and you can just ignore it.

### 4.1 Tuning

We use the boolq dataset as an example. For t5-xxl model, PT and PPT can run on at least  4 * 32G V100 GPU. FT can run on at least 16 * 32G V100 GPU.

```bash
# few-shot 32 samples
bash scripts/boolq/few-shot/ft.sh # Fine-tuning (FT)
bash scripts/boolq/few-shot/pt.sh # Prompt Tuning (PT)
bash scripts/boolq/few-shot/pt_pretrain.sh # Pre-trained Prompt Tuning (PPT)
bash scripts/boolq/few-shot/pt_uni_pretrain.sh # Unified Pre-trained Prompt Tuning (Unified PPT)

# full data
bash scripts/boolq/full/ft.sh # Fine-tuning (FT)
bash scripts/boolq/full/pt.sh # Prompt Tuning (PT)
bash scripts/boolq/full/pt_pretrain.sh # Pre-trained Prompt Tuning (PPT)
bash scripts/boolq/full/pt_uni_pretrain.sh # Unified Pre-trained Prompt Tuning (Unified PPT)
```

### 4.2 Pre-training

```bash
bash scripts/pretrain/pretrain_nsp.sh # Next Sentence Prediction
bash scripts/pretrain/pretrain_nss.sh # Next Sentence Selelction
bash scripts/pretrain/pretrain_cls.sh # Single Sentence Classificatin
bash scripts/pretrain/pretrain_nss_uni.sh # Unified Next Sentence Selelction (for Unified PPT)
```



## 5 Cite

If you use the code, please cite the following paper:

```latex
@inproceedings{gu2022ppt,
  title={PPT: Pre-trained Prompt Tuning for Few-shot Learning},
  author={Gu, Yuxian and Han, Xu and Liu, Zhiyuan and Huang, Minlie},
  booktitle={Proceedings of ACL},
  year={2022}
}
```

