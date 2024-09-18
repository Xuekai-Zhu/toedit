#!/bin/bash

python OLMo/scripts/prepare_memmap_dataset.py data/finance_all/finance_orca_1_2/*.json.gz \
    -odata/finance_all/finance_orca_1_2_tokenized \
    --tokenizer OLMo/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json \
    --workers 8