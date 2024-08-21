python OLMo/scripts/prepare_memmap_dataset.py data/math/openwebmath_lt_0.001_top_p_0.9/*.json.gz \
    -o data/math/openwebmath_lt_0.001_top_p_0.9_tokenized \
    --tokenizer OLMo/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json \
    --workers 8