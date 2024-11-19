#!/bin/bash

INPUT_DIR="/home/xkzhu/scaling_down_data/probability/sft/code-data/Magicoder-OSS-Instruct-75K-json-gz"
OUTPUT_DIR="data/sft/code-data-revised/Magicoder-OSS-Instruct-75K-json-gz"
CHUNK_SIZE=1000000  

python data_process/merge_jsonl2gz.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR" --chunk_size $CHUNK_SIZE
