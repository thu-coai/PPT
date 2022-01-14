WORKING_DIR=/mnt/sfs_turbo/gyx/PPT


LOAD_PATH="/mnt/sfs_turbo/gyx/CPM-2-Pretrain-En/results/nsp_10g_3c_en_lr0.1/8000/8000/mp_rank_00_model_states.pt"
SAVE_PATH="/mnt/sfs_turbo/gyx/checkpoints/t5-xxl/t5-MP4/1/mp_rank_00_model_states.pt"

python3 ${WORKING_DIR}/test_ckpt.py ${LOAD_PATH} ${SAVE_PATH}