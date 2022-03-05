WORKING_DIR=/home/guyuxian/PPT-origin

CKPT="nsp_10g_3c_en_lr0.1"
CKPT_ITER=8000
LOAD_PATH="/mnt/sfs_turbo/gyx/CPM-2-Pretrain-En/results/${CKPT}/${CKPT_ITER}/${CKPT_ITER}/"
SAVE_PATH="${WORKING_DIR}/prompt_embeds/pretrain-${CKPT}-${CKPT_ITER}.pt"

python3 ${WORKING_DIR}/tools/get_prompt.py ${LOAD_PATH} ${SAVE_PATH}