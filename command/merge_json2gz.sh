#!/bin/bash

INPUT_DIR="/home/xkzhu/scaling_down_data/data/less-data-llama-revised/cot-json-gz"
OUTPUT_DIR="data/less-data-llama-revised/cot-json-gz-gz"
CHUNK_SIZE=1000000

python data_process/merge_jsonl2gz.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR" --chunk_size $CHUNK_SIZE
