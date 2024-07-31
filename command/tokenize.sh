python OLMo/scripts/prepare_memmap_dataset.py data/bio/OpenOrca-1B/*.json.gz \
    -o data/bio/OpenOrca-1B_tokenized \
    --tokenizer OLMo/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json \
    --workers 4