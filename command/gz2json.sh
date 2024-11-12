for file in data/less-data/train/processed/oasst1-json-gz/*.json.gz; do
    gzip -d "$file"
done