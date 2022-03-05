#! /bin/bash

WORKING_DIR=/home/guyuxian/PPT-origin

MP_SIZE=4

NUM_GPUS_PER_WORKER=8 # number of gpus used on one node

DATA_EXT=".jsonl"
DATA_PATH="${WORKING_DIR}/downstream_data/English/sst2-full"

MASTER_PORT=${1-1234}
LR=${2-0.01}
GRAD_ACC=${3-1}
SEED=${4-10}
CKPT=${4-sentiment_10g_yelp_fix_lr0.1}
CKPT_ITER=${5-22000}

CONFIG_PATH="${WORKING_DIR}/configs/model/t5_xxl_config.json"
CKPT_PATH="${WORKING_DIR}/checkpoints/t5-xxl/t5-MP4"
PROMPT_PATH="${WORKING_DIR}/pretrained_prompts/pretrain-${CKPT}-${CKPT_ITER}.pt"

SAVE_PATH="${WORKING_DIR}/results/sst2/full/ppt/lr${LR}_G${GRAD_ACC}/seed${SEED}/"
LOG_FILE="${SAVE_PATH}/log.txt"
DS_CONFIG="${WORKING_DIR}/configs/deepspeed/ds_fp16.json"
TOKENIZER_PATH="${WORKING_DIR}/vocab_en"

PROMPT_CONFIG="${WORKING_DIR}/configs/prompt/ppt.json"

BATCH_SIZE=16
TRAIN_ITER=-1
EPOCHS=1


OPTS=""
OPTS+=" --model-config ${CONFIG_PATH}"
OPTS+=" --model-parallel-size ${MP_SIZE}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --dev-batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --train-iters ${TRAIN_ITER}"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --log-file ${LOG_FILE}"
OPTS+=" --load ${CKPT_PATH}"
OPTS+=" --load_prompt ${PROMPT_PATH}"
OPTS+=" --data-path ${DATA_PATH}"
OPTS+=" --data-ext ${DATA_EXT}"
OPTS+=" --data-name sst2"
OPTS+=" --distributed-backend nccl"
OPTS+=" --lr ${LR}"
OPTS+=" --no-load-optim"
OPTS+=" --lr-decay-style constant"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --warmup 0.0"
OPTS+=" --tokenizer-path ${TOKENIZER_PATH}"
OPTS+=" --save-interval 100000"
OPTS+=" --eval-interval 100"
OPTS+=" --log-interval 20"
OPTS+=" --checkpoint-activations"
OPTS+=" --deepspeed-activation-checkpointing"
OPTS+=" --fp16"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${DS_CONFIG}"
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --seed ${SEED}"
OPTS+=" --prompt-tune"
OPTS+=" --prompt-config ${PROMPT_CONFIG}"
OPTS+=" --epochs ${EPOCHS}"

CMD="torchrun --master_port ${MASTER_PORT} --nproc_per_node ${NUM_GPUS_PER_WORKER} ${WORKING_DIR}/train.py ${OPTS}"

echo ${CMD}
mkdir -p ${SAVE_PATH}
${CMD} 2>&1 | tee ${SAVE_PATH}/train_log
