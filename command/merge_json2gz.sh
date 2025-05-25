#!/bin/bash

INPUT_DIR="probability/sft/less-data/cot-json-gz-1"
OUTPUT_DIR="data/less-data-llama-revised-2/cot-json-gz"
CHUNK_SIZE=1000000

python data_process/merge_jsonl2gz.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR" --chunk_size $CHUNK_SIZE
