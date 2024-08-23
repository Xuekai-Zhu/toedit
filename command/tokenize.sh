python OLMo/scripts/prepare_memmap_dataset.py data/bio/biomed_8_lt_0.001/*.json.gz \
    -o data/bio/biomed_8_lt_0.001_tokenized \
    --tokenizer OLMo/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json \
    --workers 8