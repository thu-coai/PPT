#! /bin/bash

WORKING_DIR=/mnt/sfs_turbo/gyx/PPT

NUM_WORKERS=2
NUM_GPUS_PER_WORKER=8
HOST_FILE=${WORKING_DIR}/configs/host_files/hostfile-s0-s1

MP_SIZE=4

DATA_EXT=".jsonl"
DATA_PATH="/dataset/f1d6ea5b/gyx/data/lcqmc"

LR=${1-0.005}
GRAD_ACC=${2-1}
SEED=${3-1234}
NUM=${4-1024}

CONFIG_PATH="${WORKING_DIR}/configs/model/cpm2_config.json"
CKPT_PATH="/dataset/f1d6ea5b/gyx/CPM-2-dense/"

SAVE_PATH="${WORKING_DIR}/results/lcqmc/few-shot/lr${LR}_G${GRAD_ACC}_prompt_num${NUM}_init_vocab_limit1000_mantemplate/seed${SEED}/"
LOG_FILE="${SAVE_PATH}/log.txt"
DS_CONFIG="${WORKING_DIR}/configs/deepspeed/ds_full_model.json"
TOKENIZER_PATH="${WORKING_DIR}/bpe_cn"

PROMPT_CONFIG="${WORKING_DIR}/configs/prompt/lcqmc/lcqmc_5_5_0_init_vocab.json"

BATCH_SIZE=4
TRAIN_ITER=-1
EPOCHS=200


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
OPTS+=" --train-num ${NUM}"
# OPTS+=" --do-eval"
OPTS+=" --prompt-tune"
OPTS+=" --prompt-config ${PROMPT_CONFIG}"
# OPTS+=" --do_infer"
OPTS+=" --epochs ${EPOCHS}"

# CMD="python -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${WORKING_DIR}/finetune_cpm2.py ${OPTS}"
CMD="deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE} ${WORKING_DIR}/finetune_cpm2.py ${OPTS}"

# echo "Copying Data"
# cp -r ${ORIGIN_DATA_PATH} ${CACHE_PATH}
# ls ${DATA_PATH}
# echo "Copy End"

echo ${CMD}
mkdir -p ${SAVE_PATH}
${CMD} 2>&1 | tee ${SAVE_PATH}/train_log

set +x
