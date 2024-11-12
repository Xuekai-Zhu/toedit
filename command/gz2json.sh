for file in data/natural-instructions-json-gz-llama-revised/*.json.gz; do
    gzip -d "$file"
done