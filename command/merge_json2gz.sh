#!/bin/bash

INPUT_DIR="probability/math/open-web-math—1B-up_revise_Llama-3-8B-Instruct"
OUTPUT_DIR="data/math/open-web-math—1B-up_revise_Llama-3-8B-Instruct"
CHUNK_SIZE=1000000  

python data_process/merge_jsonl2gz.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR" --chunk_size $CHUNK_SIZE

python OLMo/scripts/prepare_memmap_dataset.py  "$OUTPUT_DIR"/*.json.gz \
    -o "$OUTPUT_DIR"_tokenized \
    --tokenizer OLMo/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json \
    --workers 8