#!/bin/bash

INPUT_DIR="/home/xkzhu/scaling_down_data/probability/sft/less-data/oasst1-json-gz"
OUTPUT_DIR="data/less-data-llama-revised/oasst1-json-gz"
CHUNK_SIZE=100000  

python data_process/merge_jsonl2gz.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR" --chunk_size $CHUNK_SIZE
