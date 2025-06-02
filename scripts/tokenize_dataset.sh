# Modify this to your local path, e.g., /home/ubuntu/DMoE
export MAIN_DIR="path2DMoE"
cd ${MAIN_DIR}

FILES=("3Groups-train" "3Groups-merge-train")

for FILE in "${FILES[@]}"; do
    python -u src/tokenize_dataset.py \
        -w 16 \
        -b 2048 \
        -n text \
        -t "bigscience/bloom-560m" \
        -s ./data/source/${FILE} \
        -o ./data/datasets/${FILE}
done
