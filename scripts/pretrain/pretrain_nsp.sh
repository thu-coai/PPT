#! /bin/bash

WORKING_DIR=/home/guyuxian/PPT-origin

MP_SIZE=4

NUM_GPUS_PER_WORKER=8 # number of gpus used on one node
NUM_NODES=2

DATA_PATH="${WORKING_DIR}/pretrain_data/preprocessed/nsp_document"

CONFIG_PATH="${WORKING_DIR}/configs/model/t5_xxl_config.json"
CKPT_PATH="${WORKING_DIR}/checkpoints/t5-xxl/t5-MP4"

DS_CONFIG="${WORKING_DIR}/src/configs/deepspeed/ds_fp16.json"
TOKENIZER_PATH="${WORKING_DIR}/vocab_en"

PROMPT_CONFIG="${WORKING_DIR}/src/configs/prompt/pretrain.json"

MASTER_PORT=${1-1234}
BATCH_SIZE=64
GRAD_ACC=1
LR=0.1
TRAIN_ITER=200000

SAVE_PATH="${WORKING_DIR}/results/nsp_lr${LR}_G${GRAD_ACC}/"
LOG_FILE="${SAVE_PATH}/log.txt"

ENC_LEN=512
DEC_LEN=10

OPTS=""
OPTS+=" --model-config ${CONFIG_PATH}"
OPTS+=" --model-parallel-size ${MP_SIZE}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --enc-seq-length ${ENC_LEN}"
OPTS+=" --dec-seq-length ${DEC_LEN}"
OPTS+=" --train-iters ${TRAIN_ITER}"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --log-file ${LOG_FILE}"
OPTS+=" --load ${CKPT_PATH}"
OPTS+=" --data-path ${DATA_PATH}"
OPTS+=" --data-impl mmap"
OPTS+=" --split 949,50,1"
OPTS+=" --distributed-backend nccl"
OPTS+=" --lr ${LR}"
OPTS+=" --no-load-optim"
OPTS+=" --lr-decay-style noam"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --warmup 0.00"
OPTS+=" --tokenizer-path ${TOKENIZER_PATH}"
OPTS+=" --save-interval 2000"
OPTS+=" --eval-interval 2000"
OPTS+=" --log-interval 100"
OPTS+=" --checkpoint-activations"
OPTS+=" --deepspeed-activation-checkpointing"
OPTS+=" --fp16"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${DS_CONFIG}"
OPTS+=" --prompt-tune"
OPTS+=" --prompt-config ${PROMPT_CONFIG}"
OPTS+=" --pretrain-task nsp"
OPTS+=" --save-prompt-only"

CMD="torchrun --master_port ${MASTER_PORT} --nproc_per_node ${NUM_GPUS_PER_WORKER} ${WORKING_DIR}/pretrain.py ${OPTS}"

echo ${CMD}
mkdir -p ${SAVE_PATH}
${CMD} 2>&1 | tee ${SAVE_PATH}/train_log
