for file in /home/zhuxuekai/scratch2_nlp/scaling_down_data/data/bio/biomed+orca-1B-json/biomed+orca-1B/*.json.gz; do
    gzip -d "$file"
done