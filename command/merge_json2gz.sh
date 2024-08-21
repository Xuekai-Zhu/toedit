#!/bin/bash

INPUT_DIR="probability/openwebmath_lt_0.001_top_p_0.9"
OUTPUT_DIR="data/math/openwebmath_lt_0.001_top_p_0.9"
CHUNK_SIZE=1000000  

python data_process/merge_jsonl2gz.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR" --chunk_size $CHUNK_SIZE