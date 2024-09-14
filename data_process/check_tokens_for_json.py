import os
import gzip
import json
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer


def search_files(input_dir):
    # Traverse the directory to get all sampled files
    files_to_process = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(root, file)
                files_to_process.append(file_path)
    
    return files_to_process

def statistic_tokens(input_dir):

    files_to_process = search_files(input_dir)
    tokenizer = AutoTokenizer.from_pretrained("pretrained_tokenizer/Meta-Llama-3.1-8B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    
    all_tokens = 0
    for file_path in tqdm(files_to_process, desc="Processing files"):
        with open(file_path, 'r') as f:
            lines = f.readlines()
            i_tokens = tokenizer(lines)
            token_nums = sum(len(ids) for ids in i_tokens["input_ids"])
            all_tokens += token_nums
            
    print(f"Total tokens: {all_tokens}")
    

if __name__ == '__main__':
    source_path = "probability/open-web-math—1B-up_revise_Llama-3-8B-Instruct"
    statistic_tokens(source_path)
    
    


    
