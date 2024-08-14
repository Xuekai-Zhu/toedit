import json
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import os

def list_files_in_subdirectories(parent_directory):
    all_files = []
    for root, dirs, files in os.walk(parent_directory):
        for file in files:
            if file.endswith('.jsonl'):
                all_files.append(os.path.join(root, file))
    return all_files

def load_text(file_path):
    with open(file_path, 'rt') as f:
        data = f.readlines()
        text_list = [json.loads(i) for i in data]
    return text_list


def process(in_file, tokenizer, out_file):
    data = load_text(in_file)
    
    results = []
    for entry in data:
        log_probs = entry["prompt_logprobs"]
        token_ids = entry["prompt_token_ids"]
        
        if log_probs[0] is None:
            log_probs[0] = 0
        
        probs = np.exp(log_probs)
        
        main_id = np.array(token_ids)[probs >= 0.001]
        
        tokens = tokenizer.decode(main_id.tolist())
        results.append({"tokens": tokens,})

    with open(out_file, 'w') as f:
        for i in results:
            f.write(json.dumps(i) + '\n')



if __name__ == '__main__':
    source_path = "probability/biomed"
    output_dir = "probability/after_filtering"
    all_files = list_files_in_subdirectories(source_path)
    
    tokenizer = AutoTokenizer.from_pretrained("/data1/xkzhu/pre_trained_model/Qwen/Qwen2-0.5B")
    
    for i, file_i in enumerate(tqdm(all_files)):
        outfile = os.path.join(output_dir, f"filtered_token_<0.001_{i}.jsonl")
        process(file_i, tokenizer, outfile)