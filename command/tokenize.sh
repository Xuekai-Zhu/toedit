python OLMo/scripts/prepare_memmap_dataset.py data/math/open-web-math/open-web-math—2B/*.json.gz \
    -o data/math/open-web-math/open-web-math—2B_tokenized_test \
    --tokenizer OLMo/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json \
    --workers 8