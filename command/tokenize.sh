#!/bin/bash

python OLMo/scripts/prepare_memmap_dataset.py  data/biomed+orca/biomed+orca-1B/*.json.gz \
    -o data/biomed+orca/biomed+orca-1B_tokenized \
    --tokenizer OLMo/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json \
    --workers 8