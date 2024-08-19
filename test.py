from datasets import load_dataset
from datasets import load_from_disk
from tqdm import tqdm

import gzip
import json

# all_prompts = [1, 2,3 ,4, 5, 6, 7, 8, 9, 10]
# batch_size = 2
# for start in tqdm(range(0, len(all_prompts), batch_size), desc=f"Processing batch"):
            
#         batch_prompts = all_prompts[start:start+batch_size]
#         print(batch_prompts)

def read_gz_file(path):
    """
    Read a gzipped JSON file line by line, yielding each JSON object.
    """
    with gzip.open(path, 'rt') as f:
        # for line in f:
        data = f.readlines()
    texts = []
    for line in data:
        json_line = json.loads(line)
        texts.append(json_line["text"])
    return texts

data = read_gz_file("data/math/meta-math/MetaMathQA—1B/part_0.json.gz")
print(len(data))