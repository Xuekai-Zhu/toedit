import sys
import os
from read_compre import cook_pt_entries
from tqdm import tqdm
import json
import gzip

def write_to_gz(lines, output_dir, part):
    part_file = os.path.join(output_dir, f"part_{part}.json.gz")
    with gzip.open(part_file, 'wt') as gz_file:
        for line in lines:
            item = json.dumps({"text":line})
            gz_file.write(item + "\n")
    print(f"Saved {len(lines)} lines to {part_file}")



train_json_path = "data/syn4-27_fin/114-171-of-fin_raw_news+reddit_output/json/train.json" 
output_dir = "data/finance"
chunk_size = 10000  # 调整块大小
file_count = 0

os.makedirs(output_dir, exist_ok=True)
data_paths = json.load(open(train_json_path))[0]["source"]

all_entries = []
for path in tqdm(data_paths):
    with open(path, 'r', encoding='utf8') as f:
        jsonls = f.read().strip().split('\n')
    for jsonl in jsonls:
        all_entries.append(json.loads(jsonl))

instruction_augmented_texts = []
for idx, entry in enumerate(tqdm(all_entries)):
    texts = cook_pt_entries(read_collection=entry, random_seed=idx)                              
    instruction_augmented_texts.extend(texts)
    
    while len(instruction_augmented_texts) >= chunk_size:
        write_to_gz(instruction_augmented_texts[:chunk_size], output_dir, file_count)
        instruction_augmented_texts = instruction_augmented_texts[chunk_size:]
        file_count += 1

if instruction_augmented_texts:
    write_to_gz(instruction_augmented_texts, output_dir, file_count)