import gzip
import json

def load_text(file_path):
    with gzip.open(file_path, 'rt') as f:
        data = f.readlines()
        print(len(data))
        
def load_text_in_chunks(file_path, chunk_size=1000):
    with gzip.open(file_path, 'rt') as f:
        chunk = []
        for line in f:
            chunk.append(json.loads(line))
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

def load_json_in_chunks(file_path, chunk_size=1000):
    with open(file_path, 'r') as f:
        chunk = []
        for line in f:
            chunk.append(line)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

n = 0
for chunk_id, chunk in enumerate(load_json_in_chunks("probability/biomed_8/shard_1_of_8/0_Qwen2-0.5B-Instruct_0.jsonl")):
    n += 1
    
print(n)
