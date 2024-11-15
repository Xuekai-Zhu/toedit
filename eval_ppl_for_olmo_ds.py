# -------------------- Standard Libraries -------------------- #
import os
import json
import logging
import copy
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import functools
import csv
# -------------------- Third-Party Libraries -------------------- #
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm
import wandb
from datasets import load_dataset, load_from_disk
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import deepspeed
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn import CrossEntropyLoss

# ------------------------------------- Arguments ------------------------------------- #
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Script Configuration")

    # Model Arguments
    parser.add_argument("--model_name_or_path", default="facebook/opt-125m", type=str,
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    # Add local_rank argument for DeepSpeed
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")

    # Data Arguments
    parser.add_argument("--test_file", default=None, type=str,
                        help="An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file).")
    parser.add_argument("--per_device_eval_batch_size", default=1, type=int,
                    help="Batch size for processing the test data.")
    parser.add_argument("--preprocessing_num_workers", default=None, type=int,
                        help="The number of processes to use for the preprocessing.")
    parser.add_argument("--max_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--dataset_name", default="default_dataset", type=str,
                    help="The name of the dataset to be used.")
    parser.add_argument("--num_shards", default=0, type=int, help="Total number of shards in the dataset")
    parser.add_argument("--shard_index", default=0, type=int, help="Index of the current shard being processed")
    parser.add_argument("--pre_tokenized", action='store_true', help="Whether input data is tokenized")

    # Results Arguments
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                    help="The output directory where the model predictions and checkpoints will be written.")

    args = parser.parse_args()
    return args

# ------------------------------------- Utility Functions ------------------------------------- #

def tokenizing_function(args, text_input=None, tokenizer=None):
    padding = "max_length"
    model_inputs = tokenizer(text_input, 
                            max_length=args.max_length, 
                            padding=padding,
                            truncation=True, 
                            return_tensors="pt")
    model_inputs["labels"] = torch.where(model_inputs["input_ids"] == tokenizer.pad_token_id, -100, model_inputs["input_ids"])
    
    return model_inputs

def load_eval_data():
    eval_files = eval_files = {
                                "ArXiv": "data/pile-val/subset/ArXiv.json",
                                "BookCorpus2": "data/pile-val/subset/BookCorpus2.json",
                                "Books3": "data/pile-val/subset/Books3.json",
                                "DM_Mathematics": "data/pile-val/subset/DM_Mathematics.json",
                                "Enron_Emails": "data/pile-val/subset/Enron_Emails.json",
                                "EuroParl": "data/pile-val/subset/EuroParl.json",
                                "FreeLaw": "data/pile-val/subset/FreeLaw.json",
                                "Github": "data/pile-val/subset/Github.json",
                                "Gutenberg_(PG-19)": "data/pile-val/subset/Gutenberg_(PG-19).json",
                                "HackerNews": "data/pile-val/subset/HackerNews.json",
                                "NIH_ExPorter": "data/pile-val/subset/NIH_ExPorter.json",
                                "OpenSubtitles": "data/pile-val/subset/OpenSubtitles.json",
                                "OpenWebText2": "data/pile-val/subset/OpenWebText2.json",
                                "PhilPapers": "data/pile-val/subset/PhilPapers.json",
                                "Pile-CC": "data/pile-val/subset/Pile-CC.json",
                                "PubMed_Abstracts": "data/pile-val/subset/PubMed_Abstracts.json",
                                "PubMed_Central": "data/pile-val/subset/PubMed_Central.json",
                                "StackExchange": "data/pile-val/subset/StackExchange.json",
                                "USPTO_Backgrounds": "data/pile-val/subset/USPTO_Backgrounds.json",
                                "Ubuntu_IRC": "data/pile-val/subset/Ubuntu_IRC.json",
                                "Wikipedia_(en)": "data/pile-val/subset/Wikipedia_(en).json",
                                "YoutubeSubtitles": "data/pile-val/subset/YoutubeSubtitles.json"
                            }
    all_eval_text = {}
    for i in eval_files.keys():
        with open(eval_files[i], "r") as f:
            data = f.readlines()
            data = [json.loads(j)["text"] for j in data]
            all_eval_text[i] = data
    return all_eval_text


class Eval_Dataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts
    
    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.prompts)
    def __getitem__(self, index):
        prompt = self.prompts[index]
        return prompt


def move_to_device(data, device):
    
    if isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data


def shard_list(data_list, num_shards, shard_index):
    if num_shards <= 0 or shard_index < 1 or shard_index > num_shards:
        raise ValueError("Invalid number of shards or shard index.")

    sorted_list = sorted(data_list)
    shard_size = len(sorted_list) // num_shards
    remainder = len(sorted_list) % num_shards

    start_index = 0
    shards = []

    for i in range(num_shards):
        end_index = start_index + shard_size + (1 if i < remainder else 0)
        shards.append(sorted_list[start_index:end_index])
        start_index = end_index
        
    return shards[shard_index-1]


# ------------------------------------- Main Function ------------------------------------- #

def main():
    # Argument Parsing
    args = parse_args()
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    # device = torch.device("cuda", args.local_rank)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    results_dir = os.path.join(args.output_dir, "test_ppl")
    os.makedirs(results_dir, exist_ok=True)
        
    # load eval
    eval_data = load_eval_data()    
    
    # Model Initialization
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                 local_files_only=True).to(device)
    # ds_engine = deepspeed.init_inference(model,
    #                              tensor_parallel={"tp_size": world_size},
    #                             #  dtype=torch.half,
    #                             #  checkpoint=None if args.pre_load_checkpoint else args.checkpoint_json,
    #                              replace_with_kernel_inject=True)
    # model = ds_engine.module

    # Tokenizer Initialization
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Test Loop
    model.eval()  
    with torch.no_grad(): 
        for dataset_name in eval_data.keys():
            all_loss = []
            
            all_prompts = eval_data[dataset_name]
            test_dataset = Eval_Dataset(all_prompts)
            # sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=args.local_rank, shuffle=False, drop_last=True)
            test_dataloader = DataLoader(
                            test_dataset,
                            batch_size=args.per_device_eval_batch_size, 
                            # sampler=sampler,
                            # drop_last=True,
                            )
            
            for model_inputs in tqdm(test_dataloader, desc=f"Test (Rank {args.local_rank})"):   
                model_inputs = tokenizing_function(args, text_input=model_inputs, tokenizer=tokenizer).to(device)      

                train_outputs = model(**model_inputs)
                neg_log_likelihood = train_outputs.loss
                # per_sample_loss = sample_loss(train_outputs.logits, model_inputs["labels"])
                
                # Calculate and log perplexity
                # ppls = torch.exp(per_sample_loss)
                # for ppl in ppls:
                #     all_ppls.append(ppl.item())
                # ppl = torch.exp(neg_log_likelihood).item()
                all_loss.append(neg_log_likelihood.cpu().item())
                
            # Gather all perplexities at rank 0
            # ppls_tensor = torch.tensor(all_ppls, device=device)
            # if args.local_rank == 0:    
            #     gathered_tensors = [torch.zeros_like(ppls_tensor) for _ in range(world_size)]
            # else:
            #     gathered_tensors = None
            # dist.gather(ppls_tensor, gather_list=gathered_tensors, dst=0)
            
            # save 
            # if args.local_rank == 0:
                # Flatten list of lists
            stabilized_loss = min(20, np.mean(all_loss))
            ppl = np.exp(stabilized_loss)
            results_file = os.path.join(results_dir, f"{dataset_name}_ppl.csv")
            # gathered_ppls = [item for tensor in gathered_tensors for item in tensor.cpu().tolist()]
            with open(results_file, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['PPL'])
                # for ppl in gathered_ppls:
                csvwriter.writerow([ppl])
                
                
                
                
            
if __name__ == "__main__":
    main()