import json
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import gzip
import multiprocessing


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


def process_file(in_files, tokenizer, out_dir, threshold, process_id):
    
    for i, file_i in enumerate(tqdm(in_files)):
        data = load_text(file_i)
        results = []
        for entry in data:
            log_probs = entry["prompt_logprobs"]
            token_ids = entry["prompt_token_ids"]
            
            if log_probs[0] is None:
                log_probs[0] = 0
            
            probs = np.exp(log_probs)
            
            main_id = np.array(token_ids)[probs >= threshold]
            # main_id = np.array(token_ids)[probs <= threshold]
            
            tokens = tokenizer.decode(main_id.tolist())
            results.append({"text": tokens,})
            
        out_file = os.path.join(out_dir, f"process_{process_id}_file_{i}_filtered_token_<{threshold}.jsonl")
        with open(out_file, 'w') as f:
            for i in results:
                f.write(json.dumps(i) + '\n')

  

def main_step1(num_processes, source_path, output_dir, strategy=None, threshold=None):
    
    os.makedirs(output_dir, exist_ok=True)
    all_files = list_files_in_subdirectories(source_path)
    tokenizer = AutoTokenizer.from_pretrained("pre_trained_model/Qwen2-0.5B-Instruct")
    file_chunks = [all_files[i::num_processes] for i in range(num_processes)]

    for i, file_chunk in enumerate(file_chunks):
        if strategy == "filter":
            # strategy_1_filter(file_chunk, tokenizer, output_dir, threshold, i)
            process = multiprocessing.Process(target=strategy_1_filter, args=(file_chunk, tokenizer, output_dir, threshold, i))
            process.start()


def strategy_1_filter(in_files, tokenizer, out_dir, threshold, process_id):
    for i, file_i in enumerate(tqdm(in_files)):
        data = load_text(file_i)
        results = []
        for entry in data:
            log_probs = entry["prompt_logprobs"]
            token_ids = entry["prompt_token_ids"]
            
            processed_log_probs = []
            for probs, id in zip(log_probs, token_ids):
                if probs is None:
                    processed_log_probs.append(0)
                else:
                    processed_log_probs.append(probs[f"{id}"]["logprob"])
            
            probs = np.exp(processed_log_probs)
            
            main_id = np.array(token_ids)[probs >= threshold]
            # main_id = np.array(token_ids)[probs <= threshold]
            
            tokens = tokenizer.decode(main_id.tolist())
            results.append({"text": tokens,})
            
        out_file = os.path.join(out_dir, f"process_{process_id}_file_{i}_filtered_token_<{threshold}.jsonl")
        with open(out_file, 'w') as f:
            for i in results:
                f.write(json.dumps(i) + '\n')
    

if __name__ == '__main__':
    num_processes = 8
    
    # open web math
    strategy = "filter"
    threshold = 0.001
    source_path = "probability/openwebmath"
    output_dir = f"probability/openwebmath<{threshold}"
    
    main_step1(num_processes, source_path, output_dir, strategy=strategy, threshold=threshold)
      
    # for i, file_i in enumerate(tqdm(all_files)):
    #     outfile = os.path.join(output_dir, f"filtered_token_<0.001_{i}.jsonl")
    #     process(file_i, tokenizer, outfile)