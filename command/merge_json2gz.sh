#!/bin/bash

INPUT_DIR="probability/biomed_8_Meta-Llama3-8B-Instruct_gt_0.99"
OUTPUT_DIR="data/bio/biomed_8_Meta-Llama3-8B-Instruct_gt_0.99_resmapling"
CHUNK_SIZE=1000000  

python data_process/merge_jsonl2gz.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR" --chunk_size $CHUNK_SIZE