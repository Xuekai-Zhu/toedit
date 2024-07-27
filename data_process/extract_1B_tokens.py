import os
import gzip
import json
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

def merge_sampled_files(input_dir, output_dir, chunk_size=1000000, target_tokens=1e9):
    all_sampled_lines = []
    file_count = 0
    total_tokens = 0
    
    # 加载数据集
    dataset = load_dataset(input_dir, cache_dir=f"{input_dir}/cache", split="train")
    dataset_size = len(dataset)
    print(f"Dataset size: {dataset_size}")
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained("pre_trained_model/step738020-unsharded-hf")
    
    # 采样前1000条数据计算token数量
    if "Arithmo-Data" in input_dir:
        sample_lines = [item["question"] + " " + item["answer"] for item in dataset.select(range(1000))]
    elif "MetaMathQA" in input_dir:
        sample_lines = [item["query"] + " " + item["response"] for item in dataset.select(range(1000))]
        
    sample_tokens = sum([len(tokenizer.tokenize(text)) for text in sample_lines])
    avg_tokens_per_line = sample_tokens / 1000
    
    # 计算所需行数
    required_lines = int(target_tokens / avg_tokens_per_line)
    
    if required_lines >= dataset_size:
        required_lines = dataset_size
        selected_datasets = dataset
    else:
        selected_datasets = dataset.select(range(required_lines))
        
    print(f"Estimated number of lines needed: {required_lines}")
    
    
    for item in tqdm(selected_datasets):
        if "Arithmo-Data" in input_dir:
            text_line = json.dumps({"text": item["question"] + " " + item["answer"]}) + "\n"
        elif "MetaMathQA" in input_dir:
            text_line = json.dumps({"text": item["query"] + " " + item["response"]}) + "\n"
            
        all_sampled_lines.append(text_line)
        # total_tokens += len(tokenizer.tokenize(item["question"] + item["answer"]))
                
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
    input_dir = "data/akjindal53244/Arithmo-Data"
    output_dir = "data/akjindal53244/Arithmo-Data—1B"
    chunk_size = 1000000  # Adjust the chunk size as needed
    target_tokens = 1e9  # Target token count: 1 billion
    os.makedirs(output_dir, exist_ok=True)
    merge_sampled_files(input_dir, output_dir, chunk_size, target_tokens)
