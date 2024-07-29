python OLMo/scripts/prepare_memmap_dataset.py data/open-web-math/open-web-math—1B/*.json.gz \
    -o data/open-web-math/open-web-math—1B_tokenized \
    --tokenizer OLMo/olmo_data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json \
    --workers 4