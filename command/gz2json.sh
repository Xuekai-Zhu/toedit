for file in data/sft/code-data-revised/Magicoder-OSS-Instruct-75K-json-gz/*.json.gz; do
    gzip -d "$file"
done