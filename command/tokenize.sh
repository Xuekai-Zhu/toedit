#!/bin/bash

python OLMo/scripts/prepare_memmap_dataset.py  data/bio/biomed_gt_0.99_top_p/*.json.gz \
    -o data/bio/biomed_gt_0.99_top_p_tokenized \
    --tokenizer OLMo/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json \
    --workers 8