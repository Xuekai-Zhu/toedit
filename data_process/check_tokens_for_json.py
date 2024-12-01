import os
import gzip
import json
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer
import gzip


def search_files(input_dir):
    # Traverse the directory to get all sampled files
    files_to_process = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith((".jsonl",".gz", ".json")):
                file_path = os.path.join(root, file)
                files_to_process.append(file_path)
    
    return files_to_process

def statistic_tokens(input_dir):

    files_to_process = search_files(input_dir)
    tokenizer = AutoTokenizer.from_pretrained("/data1/xkzhu/pre_trained_model/meta-llama/Meta-Llama-3-8B")
    tokenizer.pad_token = tokenizer.eos_token
    
    all_tokens = 0
    texts = []
    for file_path in tqdm(files_to_process, desc="Processing files"):
        if file_path.endswith(".json"):
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for i in lines:
                    item = json.loads(i)
                    # text = item["input"] + " " + item["output"]
                    text = item["output"]
                    texts.append(text)

        elif file_path.endswith(".gz"):
            with gzip.open(file_path, 'rt') as f:
                lines = f.readlines()
                texts = [json.loads(line)["text"] for line in lines]
            
        i_tokens = tokenizer(texts)
        token_nums = sum(len(ids) for ids in i_tokens["input_ids"])
        all_tokens += token_nums
            
    # print(f"Total tokens: {all_tokens}")
    print(f"Total tokens: {all_tokens:,} tokens")

    

if __name__ == '__main__':
    # source_path = "probability/open-web-math—1B-up_revise_Llama-3-8B-Instruct"
    source_path = "data/less-data/less-json/cot-json-gz"
    statistic_tokens(source_path)
    
    


    
