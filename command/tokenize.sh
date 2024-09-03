#!/bin/bash

python OLMo/scripts/prepare_memmap_dataset.py  data/bio/instruction_biomed—1B-up_revise/*.json.gz \
    -o data/bio/instruction_biomed—1B-up_revise_tokenized \
    --tokenizer OLMo/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json \
    --workers 8