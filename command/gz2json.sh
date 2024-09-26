for file in data/bio/biomed_revised+orca/*.json.gz; do
    gzip -d "$file"
done