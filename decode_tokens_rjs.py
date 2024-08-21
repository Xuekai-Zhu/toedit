import json
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import gzip
import multiprocessing
import torch
import random
import argparse



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

def load_text_in_chunks(file_path, chunk_size=1000):
    with open(file_path, 'rt') as f:
        chunk = []
        for line in f:
            chunk.append(json.loads(line))
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

def main_step1(num_processes, source_path, output_dir, strategy=None, threshold=None, top_p=None):
    
    os.makedirs(output_dir, exist_ok=True)
    all_files = list_files_in_subdirectories(source_path)
    tokenizer = AutoTokenizer.from_pretrained("pre_trained_model/Qwen2-0.5B-Instruct")
    file_chunks = [all_files[i::num_processes] for i in range(num_processes)]

    for i, file_chunk in enumerate(file_chunks):
        if strategy == "filter":
            # strategy_1_filter(file_chunk, tokenizer, output_dir, threshold, i)
            process = multiprocessing.Process(target=strategy_1_filter, args=(file_chunk, tokenizer, output_dir, threshold, i))
            process.start()
        elif strategy == "top_p":
            # strategy_2_top_p(all_files, tokenizer, output_dir, threshold, i, top_p)
            process = multiprocessing.Process(target=strategy_2_top_p, args=(file_chunk, tokenizer, output_dir, threshold, i, top_p))
            process.start()


def strategy_1_filter(in_files, tokenizer, out_dir, threshold, process_id):
    # for i, file_i in enumerate(tqdm(in_files)):
    for i, file_i in enumerate(tqdm(in_files, desc="Processing files")):
        for chunk_id, chunk in enumerate(load_text_in_chunks(file_i)):
            # data = load_text(chunk)
            results = []
            # for entry in chunk:
            for entry in tqdm(chunk, desc=f"Processing entries in chunk {chunk_id}"):
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
                
            random_number = random.randint(1000, 9999)
            out_file = os.path.join(out_dir, f"process_{process_id}_file_{i}_chunk_{chunk_id}_filtered_token_lt_{threshold}_{random_number}.jsonl")
            with open(out_file, 'w') as f:
                for result in results:
                    f.write(json.dumps(result) + '\n')
    

def strategy_2_top_p(in_files, tokenizer, out_dir, threshold, process_id, top_p):
    
    def top_p_sampling(input_ids, scores, top_p, min_tokens_to_keep=1):
        filter_value = -float("Inf")
        input_ids = torch.tensor(input_ids).long()
        scores = torch.tensor(scores).float()
        
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        cumulative_probs = torch.exp(sorted_logits).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., -min_tokens_to_keep :] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
        scores_processed = scores.masked_fill(indices_to_remove, filter_value)
        p_processed = torch.softmax(scores_processed, dim=-1)
        # p_processed = torch.exp(scores_processed)
        
        next_token_index = torch.multinomial(p_processed, num_samples=1)
        next_token = input_ids[next_token_index]
        
        return int(next_token.item())
    
    for i, file_i in enumerate(tqdm(in_files, desc="Processing files")):
        for chunk_id, chunk in enumerate(load_text_in_chunks(file_i)):
        # data = load_text(file_i)
            results = []
            for entry in tqdm(chunk, desc=f"Processing entries in chunk {chunk_id}"):
                log_probs = entry["prompt_logprobs"]
                token_ids = entry["prompt_token_ids"]
                
                processed_token_ids = []
                for probs, id in zip(log_probs, token_ids):
                    if probs is None:
                        processed_token_ids.append(id)
                    else:
                        p = np.exp(probs[f"{id}"]["logprob"])

                        if p < threshold:
                            candicant_ids = [int(i) for i in probs.keys()]
                            scores = [info['logprob'] for info in probs.values()]
                            re_sampled_id = top_p_sampling(candicant_ids, scores, top_p)
                            processed_token_ids.append(re_sampled_id)
                        else:
                            processed_token_ids.append(id)
                    
                tokens = tokenizer.decode(processed_token_ids)
                results.append({"text": tokens,})
            
            random_number = random.randint(1000, 9999)
            out_file = os.path.join(out_dir, f"process_{process_id}_file_{i}_chunk_{chunk_id}_filtered_token_lt_{threshold}_{random_number}.jsonl")
            with open(out_file, 'w') as f:
                for result in results:
                    f.write(json.dumps(result) + '\n')
            
            # del results, chunk, log_probs, token_ids, processed_token_ids
                
if __name__ == '__main__':
    num_processes = 8
    
    # bio
    # strategy = "filter"
    # threshold = 0.001
    # source_path = "probability/biomed_8"
    # output_dir = f"probability/biomed_8<{threshold}"
    
    # main_step1(num_processes, source_path, output_dir, strategy=strategy, threshold=threshold)
    
    # strategy = "top_p"
    # threshold = 0.001
    # top_p = 0.9
    # source_path = "probability/biomed_8"
    # output_dir = f"probability/biomed_8_filtering/lt_{threshold}_top_p_0.9"
    # # output_dir = f"test/lt_{threshold}_top_p_0.9"
    
    # main_step1(num_processes, source_path, output_dir, strategy=strategy, threshold=threshold, top_p=top_p)
    
    # open web math
    # strategy = "filter"
    # threshold = 0.001
    # source_path = "probability/openwebmath"
    # output_dir = f"probability/openwebmath_lt_{threshold}"
    
    # main_step1(num_processes, source_path, output_dir, strategy=strategy, threshold=threshold)
    
    parser = argparse.ArgumentParser(description='Process some files with different strategies.')
    parser.add_argument('--num_processes', type=int, default=8, help='Number of processes to use')
    parser.add_argument('--source_path', type=str, required=True, help='Path to the source files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output files')
    parser.add_argument('--strategy', type=str, choices=['filter', 'top_p'], required=True, help='Strategy to use (filter or top_p)')
    parser.add_argument('--threshold', type=float, required=True, help='Threshold for filtering')
    parser.add_argument('--top_p', type=float, default=None, help='Top-p value (only required if strategy is top_p)')
    
    args = parser.parse_args()
    main_step1(
        num_processes=args.num_processes,
        source_path=args.source_path,
        output_dir=args.output_dir,
        strategy=args.strategy,
        threshold=args.threshold,
        top_p=args.top_p
    )
    