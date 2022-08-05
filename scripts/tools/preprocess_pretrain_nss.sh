WOKRING_DIR=/home/guyuxian/PPT-origin/
INPUT_PATH=${WOKRING_DIR}/pretrain_data/raw/


OPTS=""
OPTS+=" --input ${INPUT_PATH}/openwebtext.txt"
OPTS+=" --input_neg1 ${INPUT_PATH}/openwebtext_1.txt"
OPTS+=" --input_neg2 ${INPUT_PATH}/openwebtext_2.txt"
OPTS+=" --input_neg3 ${INPUT_PATH}/openwebtext_3.txt"
OPTS+=" --input_neg4 ${INPUT_PATH}/openwebtext_4.txt"
OPTS+=" --tokenizer_path ${WORKING_DIR}/vocab_en"
OPTS+=" --output_path ${WOKRING_DIR}/pretrain_data/preprocessed/"


shuf ${INPUT_PATH}/openwebtext.txt -o ${INPUT_PATH}/openwebtext_1.txt
shuf ${INPUT_PATH}/openwebtext.txt -o ${INPUT_PATH}/openwebtext_2.txt
shuf ${INPUT_PATH}/openwebtext.txt -o ${INPUT_PATH}/openwebtext_3.txt
shuf ${INPUT_PATH}/openwebtext.txt -o ${INPUT_PATH}/openwebtext_4.txt
python3 ${WOKRING_DIR}/tools/preprocess_pretrain_data_nss_uni.py ${OPTS}