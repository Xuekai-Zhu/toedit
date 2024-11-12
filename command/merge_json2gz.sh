#!/bin/bash

INPUT_DIR="/home/zhuxuekai/scaling_down_data/probability/natural-instructions-json-gz"
OUTPUT_DIR="data/natural-instructions-json-gz-llama-revised"
CHUNK_SIZE=100000  

python data_process/merge_jsonl2gz.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR" --chunk_size $CHUNK_SIZE
