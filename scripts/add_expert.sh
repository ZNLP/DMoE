# Modify this to your local path, e.g., /home/ubuntu/DMoE
export MAIN_DIR="path2DMoE"
cd ${MAIN_DIR}

MODEL_TYPE=bloom
SRC_PATH="./log/BloomMoEDyn/560M-EPS0.6-EXP9-S2_SD8/checkpoint-1500"
INDEX=1
OUT_PATH="./data/BloomMoEDyn/560M-EPS0.6-EXP9-ADD1"

python src/add_expert.py \
    -m ${MODEL_TYPE} \
    -s ${SRC_PATH} \
    -i ${INDEX} \
    -o ${OUT_PATH}