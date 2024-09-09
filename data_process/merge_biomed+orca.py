import os
import gzip
import json
import random
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np

def load_gz(in_file):
    with gzip.open(in_file, 'rt') as gz_file:
        lines = gz_file.readlines()
    return lines

def sample_fixed_tokens(sample_lines, sampled_num=1000, target_tokens=5e8):
    dataset_size = len(sample_lines)
    tokenizer = AutoTokenizer.from_pretrained("pre_trained_model/step738020-unsharded-hf")
    
    sampled_lines = random.sample(sample_lines, sampled_num)
    sample_tokens = sum([len(tokenizer.tokenize(json.loads(text)["text"])) for text in sampled_lines])
    avg_tokens_per_line = sample_tokens / sampled_num
    
    # 计算所需行数
    required_lines = int(target_tokens / avg_tokens_per_line)
    
    if required_lines >= dataset_size:
        required_lines = dataset_size
        selected_datasets = sample_lines
    else:
        selected_datasets = random.sample(sample_lines, required_lines)
        
    print(f"Estimated number of lines needed: {required_lines}")
    
    return selected_datasets

def merge_sampled_files(input_file_list, output_dir, chunk_size=1000000):
    all_sampled_lines = []
    file_count = 0

    # 读取所有文件中的所有行
    for input_files in input_file_list:
        one_source_lines = []
        for file_path in tqdm(input_files):
            lines = load_gz(file_path)
            one_source_lines.extend(lines)
            
        selected_datasets = sample_fixed_tokens(one_source_lines)
        all_sampled_lines.extend(selected_datasets)
    
    # 打乱所有行的顺序
    random.shuffle(all_sampled_lines)
    
    # 按块大小写入文件
    while len(all_sampled_lines) >= chunk_size:
        write_to_gz(all_sampled_lines[:chunk_size], output_dir, file_count)
        all_sampled_lines = all_sampled_lines[chunk_size:]
        file_count += 1
    
    # 写入剩余的行
    if all_sampled_lines:
        write_to_gz(all_sampled_lines, output_dir, file_count)
    
    print(f"Merged {file_count} files into {output_dir}")

def write_to_gz(lines, output_dir, part):
    part_file = os.path.join(output_dir, f"part_{part}.json.gz")
    with gzip.open(part_file, 'wt') as gz_file:
        for line in lines:
            gz_file.write(line)
    print(f"Saved {len(lines)} lines to {part_file}")

def search_datasets(in_path):
    all_train_files = []
    for filename in os.listdir(in_path):
        if filename.endswith('.gz'):
            file_path = os.path.join(in_path, filename)
            all_train_files.append(file_path)
    return all_train_files

if __name__ == "__main__":
    source_1 = "data/bio/biomed_context"
    source_files_1 = search_datasets(source_1)
    
    source_2 = "data/bio/OpenOrca-1B"
    source_files_2 = search_datasets(source_2)
    
    output_dir = "data/biomed_context+orca"
    chunk_size = 100000  # 调整块大小
    os.makedirs(output_dir, exist_ok=True)
    merge_sampled_files([source_files_1, source_files_2], output_dir, chunk_size)
