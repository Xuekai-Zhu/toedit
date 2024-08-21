#!/bin/bash


NUM_PROCESSES=8
THRESHOLD=0.001
TOP_P=0.9
SOURCE_PATH="probability/biomed_8"
OUTPUT_DIR="probability/biomed_8_filtering/lt_${THRESHOLD}_top_p_${TOP_P}"

python decode_tokens_rjs.py \
    --num_processes $NUM_PROCESSES \
    --source_path $SOURCE_PATH \
    --output_dir $OUTPUT_DIR \
    --strategy top_p \
    --threshold $THRESHOLD \
    --top_p $TOP_P