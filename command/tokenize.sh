#!/bin/bash

python OLMo/scripts/prepare_memmap_dataset.py  data/finance/*.json.gz \
    -o data/finance_tokenized \
    --tokenizer OLMo/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json \
    --workers 8