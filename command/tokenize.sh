#!/bin/bash

python OLMo/scripts/prepare_memmap_dataset.py  data/bio/biomed_8_Meta-Llama3-8B-Instruct_gt_0.99_resmapling/*.json.gz \
    -o data/bio/biomed_8_Meta-Llama3-8B-Instruct_gt_0.99_resmapling_tokenized \
    --tokenizer OLMo/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json \
    --workers 8