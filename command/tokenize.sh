python OLMo/scripts/prepare_memmap_dataset.py data/bio/biomed_delete_less_than_0.001_token/*.json.gz \
    -o data/bio/biomed_delete_less_than_0.001_token_tokenized \
    --tokenizer OLMo/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json \
    --workers 4