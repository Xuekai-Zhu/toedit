#!/bin/bash

INPUT_DIR="probability/biomed_8_filtering/biomed_8_lt_0.001"
OUTPUT_DIR="data/bio/biomed_8_lt_0.001"
CHUNK_SIZE=1000000  

python data_process/merge_jsonl2gz.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR" --chunk_size $CHUNK_SIZE