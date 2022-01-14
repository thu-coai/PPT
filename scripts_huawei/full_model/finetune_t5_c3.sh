#! /bin/bash

WORKING_DIR=/mnt/sfs_turbo/gyx/PPT

NUM_WORKERS=2
NUM_GPUS_PER_WORKER=8
HOST_FILE=${WORKING_DIR}/configs/host_files/hostfile-s0-s1

MP_SIZE=4

DATA_EXT=".json"
DATA_PATH="/mnt/sfs_turbo/data/CLUE/c3"

LR=${1-0.000003}
GRAD_ACC=${2-16}
SEED=${3-1234}

CONFIG_PATH="${WORKING_DIR}/configs/model/cpm2_config_drop.json"
CKPT_PATH="/dataset/f1d6ea5b/gyx/CPM-2-dense/"

SAVE_PATH="${WORKING_DIR}/results/c3/t5_finetune_lr${LR}const_G${GRAD_ACC}/seed${SEED}/"
LOG_FILE="${SAVE_PATH}/log.txt"
DS_CONFIG="${WORKING_DIR}/configs/deepspeed/ds_full_model.json"
TOKENIZER_PATH="${WORKING_DIR}/vocab_cn"

BATCH_SIZE=4
TRAIN_ITER=-1
EPOCHS=20


OPTS=""
OPTS+=" --model-config ${CONFIG_PATH}"
OPTS+=" --model-parallel-size ${MP_SIZE}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --train-iters ${TRAIN_ITER}"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --log-file ${LOG_FILE}"
OPTS+=" --load ${CKPT_PATH}"
OPTS+=" --data-path ${DATA_PATH}"
OPTS+=" --data-ext ${DATA_EXT}"
OPTS+=" --data-name c3"
OPTS+=" --distributed-backend nccl"
OPTS+=" --lr ${LR}"
OPTS+=" --no-load-optim"
OPTS+=" --lr-decay-style constant"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --warmup 0.0"
OPTS+=" --tokenizer-path ${TOKENIZER_PATH}"
OPTS+=" --save-interval 100000"
OPTS+=" --eval-interval 50"
OPTS+=" --eval-iters 10"
OPTS+=" --log-interval 10"
OPTS+=" --checkpoint-activations"
OPTS+=" --deepspeed-activation-checkpointing"
OPTS+=" --fp16"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${DS_CONFIG}"
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --seed ${SEED}"
# OPTS+=" --do-eval"
# OPTS+=" --do_infer"
OPTS+=" --epochs ${EPOCHS}"

# CMD="python -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${WORKING_DIR}/train.py ${OPTS}"
CMD="deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE} ${WORKING_DIR}/train.py ${OPTS}"

echo ${CMD}
mkdir -p ${SAVE_PATH}
${CMD} 2>&1 | tee ${SAVE_PATH}/train_log
