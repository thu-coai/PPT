CMD="python3 /mnt/sfs_turbo/gyx/PPT/tools/check.py"

echo ${CMD}
mkdir -p ${SAVE_PATH}
${CMD} 2>&1 | tee /mnt/sfs_turbo/gyx/PPT/debug
