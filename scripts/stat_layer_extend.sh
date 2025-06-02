# Modify this to your local path, e.g., /home/ubuntu/DMoE
export MAIN_DIR="path2DMoE"
cd ${MAIN_DIR}

# If it does not exist, you can run ./src/delta_layer/bloom.py to obtain this file.
STAT_FILE="./log/delta-stat/bloom-560m.json"
# STAT_FILE="./log/delta-stat/bloom-1.7b.json"
EPSILON=0.6

python -u ./log/delta-stat/stat.py \
    --statistic-file ${STAT_FILE} \
    --epsilon ${EPSILON}