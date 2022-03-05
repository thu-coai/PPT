WOKRING_DIR=/home/guyuxian/PPT-origin/


OPTS=""
OPTS+=" --input ${WOKRING_DIR}/pretrain_data/raw/openwebtext.txt"
OPTS+=" --tokenizer_path ${WORKING_DIR}/vocab_en"
OPTS+=" --output_path ${WOKRING_DIR}/pretrain_data/preprocessed/"


python3 ${WOKRING_DIR}tools/preprocess_pretrain_data_cls.py ${OPTS}