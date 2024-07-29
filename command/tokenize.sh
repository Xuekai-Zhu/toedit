python OLMo/scripts/prepare_memmap_dataset.py data/bio/biomed+orca/*.json.gz \
    -o data/bio/biomed+orca_tokenized \
    --tokenizer OLMo/olmo_data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json \
    --workers 4