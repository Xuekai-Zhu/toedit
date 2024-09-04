#!/bin/bash

INPUT_DIR="probability/biomed_8_Meta-Llama3-8B-Instruct_gt_0.99_beta"
OUTPUT_DIR="data/bio/biomed_8_Meta-Llama3-8B-Instruct_gt_0.99_beta_2"
CHUNK_SIZE=1000000  

python data_process/merge_jsonl2gz.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR" --chunk_size $CHUNK_SIZE

python OLMo/scripts/prepare_memmap_dataset.py  "$OUTPUT_DIR"/*.json.gz \
    -o "$OUTPUT_DIR"_tokenized \
    --tokenizer OLMo/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json \
    --workers 8