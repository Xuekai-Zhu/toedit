import os
import gzip
import json
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

# def read_gz_file(path):
#     """
#     Read a gzipped JSON file line by line, yielding each JSON object.
#     """
#     with gzip.open(path, 'rt') as f:
#         # for line in f:
#         data = f.readlines()
#     texts = []
#     for line in data:
#         json_line = json.loads(line)
#         texts.append(json_line["text"])
#     return texts

def search_datasets(test_file):
    all_train_files = []
    for filename in os.listdir(test_file):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(test_file, filename)
            all_train_files.append(file_path)
    return all_train_files

def merge_sampled_files(input_dir, output_dir, chunk_size=1000000, target_tokens=1e9):
    all_sampled_lines = []
    file_count = 0
    
    all_train_files = search_datasets(input_dir)
    assert len(all_train_files) == 1
    with open(all_train_files[0], "r") as f_in:
        all_data = f_in.readlines()
        # all_data = json.load(f_in)
    
    for item in tqdm(all_data):
        text_dict = json.loads(item)
        transformed_data = {"input": text_dict["instruction"],
                            "output": text_dict["response"]}
        text_line = json.dumps(transformed_data) + "\n"
        all_sampled_lines.append(text_line)

        if len(all_sampled_lines) >= chunk_size:
            write_to_gz(all_sampled_lines, output_dir, file_count)            
            all_sampled_lines = []
            file_count += 1

    if all_sampled_lines:
        write_to_gz(all_sampled_lines, output_dir, file_count)
    
    print(f"Merged {file_count+1} files into {output_dir}")
    # print(f"Total tokens: {total_tokens}")

def write_to_gz(lines, output_dir, part):
    part_file = os.path.join(output_dir, f"part_{part}.json.gz")
    with gzip.open(part_file, 'wt') as gz_file:
        for line in lines:
            gz_file.write(line)
    print(f"Saved {len(lines)} lines to {part_file}")

if __name__ == "__main__":
    input_dir = "data/sft/ise-uiuc/Magicoder-Evol-Instruct-110K"
    output_dir = "data/sft/ise-uiuc/Magicoder-Evol-Instruct-110K-json-gz"
    chunk_size = 100000  # Adjust the chunk size as needed
    target_tokens = 1e9  # Target token count: 1 billion
    os.makedirs(output_dir, exist_ok=True)
    merge_sampled_files(input_dir, output_dir, chunk_size, target_tokens)
