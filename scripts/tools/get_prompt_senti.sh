WORKING_DIR=/home/guyuxian/PPT-origin

CKPT="sentiment_10g_yelp_fix_lr0.1"
CKPT_ITER=22000
LOAD_PATH="/mnt/sfs_turbo/gyx/CPM-2-Pretrain-En/results/${CKPT}/${CKPT_ITER}/${CKPT_ITER}/"
SAVE_PATH="${WORKING_DIR}/prompt_embeds/pretrain-${CKPT}-${CKPT_ITER}.pt"

python3 ${WORKING_DIR}/tools/get_prompt.py ${LOAD_PATH} ${SAVE_PATH}