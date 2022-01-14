#! /bin/bash

WORKING_DIR=/mnt/sfs_turbo/gyx/PPT

if [[ $DLS_TASK_NUMBER == 1 ]]; then
    MASTER_ADDR=localhost
    MASTER_PORT=6000
    NNODES=1
    NODE_RANK=0
else
    MASTER_HOST="$BATCH_CUSTOM0_HOSTS"
    MASTER_ADDR="${MASTER_HOST%%:*}"
    MASTER_PORT="${MASTER_HOST##*:}"
    NNODES="$DLS_TASK_NUMBER"
    NODE_RANK="$DLS_TASK_INDEX"
fi

GPUS_PER_NODE=8
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

OPTIONS_NCCL="NCCL_DEBUG=info"

MP_SIZE=4

DATA_EXT=".jsonl"
DATA_PATH="/mnt/sfs_turbo/gyx/data/lcqmc/"

LR=${1-0.000002}
GRAD_ACC=${2-1}
SEED=${3-10}
NUM=${4-64}
CKPT=${5-nsp_1g_no_end_3c_lr0.1}
CKPT_ITER=${6-102000}

CONFIG_PATH="${WORKING_DIR}/configs/model/cpm2_config.json"
CKPT_PATH="/mnt/sfs_turbo/gyx/CPM-2-Pretrain/results/${CKPT}/${CKPT_ITER}"

SAVE_PATH="${WORKING_DIR}/results/lcqmc/gen_temp/prompt_from_pretrain_${CKPT}_${CKPT_ITER}/"
LOG_FILE="${SAVE_PATH}/log.txt"
DS_CONFIG="${WORKING_DIR}/configs/deepspeed/ds_full_model.json"
TOKENIZER_PATH="${WORKING_DIR}/bpe_cn"

PROMPT_CONFIG="${WORKING_DIR}/configs/prompt/lcqmc/lcqmc_10_0_0_pretrain.json"

BATCH_SIZE=32
TRAIN_ITER=-1
EPOCHS=10


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
OPTS+=" --data-name lcqmc"
OPTS+=" --distributed-backend nccl"
OPTS+=" --lr ${LR}"
OPTS+=" --no-load-optim"
OPTS+=" --lr-decay-style constant"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --warmup 0.0"
OPTS+=" --tokenizer-path ${TOKENIZER_PATH}"
OPTS+=" --save-interval 1000000"
OPTS+=" --eval-interval 100"
OPTS+=" --eval-iters 10"
OPTS+=" --log-interval 10"
OPTS+=" --checkpoint-activations"
OPTS+=" --deepspeed-activation-checkpointing"
OPTS+=" --fp16"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${DS_CONFIG}"
OPTS+=" --seed ${SEED}"
OPTS+=" --prompt-tune"
OPTS+=" --prompt-config ${PROMPT_CONFIG}"
OPTS+=" --epochs ${EPOCHS}"
OPTS+=" --train-num ${NUM}"

CMD="python -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${WORKING_DIR}/template_gen.py ${OPTS}"
# CMD="deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE} ${WORKING_DIR}/template_gen.py ${OPTS}"

echo ${CMD}
mkdir -p ${SAVE_PATH}
${CMD} 2>&1 | tee ${SAVE_PATH}/train_log
