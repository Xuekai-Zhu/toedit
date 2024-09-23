#!/bin/bash

INPUT_DIR="probability/instruction_biomed—1B-up_revise_Llama-3-8B-Instruct"
OUTPUT_DIR="data/bio/instruction_biomed—1B-up_revise_Llama-3-8B-Instruct"
CHUNK_SIZE=1000000  

python data_process/merge_jsonl2gz.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR" --chunk_size $CHUNK_SIZE
