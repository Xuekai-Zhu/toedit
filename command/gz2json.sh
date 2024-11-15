for file in data/less-data-llama-revised/oasst1-json-gz/*.json.gz; do
    gzip -d "$file"
done