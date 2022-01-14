ROOT_DIR=/root/mnt/back-folder/CPM-Finetune-gyx/results/poem_mc/t5_finetune_lr0.05const_G2_prompt_seed1234_save

CKPT=${1}

mkdir ${ROOT_DIR}/${CKPT}/${CKPT}
mv ${ROOT_DIR}/${CKPT}/mp* ${ROOT_DIR}/${CKPT}/${CKPT}/

echo ${CKPT} > ${ROOT_DIR}/${CKPT}/latest
echo ${CKPT} > ${ROOT_DIR}/${CKPT}/latest_checkpointed_iteration.txt
