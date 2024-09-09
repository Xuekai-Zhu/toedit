#!/bin/bash

python OLMo/scripts/prepare_memmap_dataset.py  data/biomed_context+orca/*.json.gz \
    -o data/biomed_context+orca_tokenized \
    --tokenizer OLMo/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json \
    --workers 8