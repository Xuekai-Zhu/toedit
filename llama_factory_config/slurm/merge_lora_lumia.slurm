#!/bin/bash

#SBATCH --account=xkzhu
#SBATCH --job-name=lora
#SBATCH --partition=ADA6000      
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=11 # 若多卡多进程，请调整此参数
#SBATCH --cpus-per-task=10  # 每个进程的CPU数量
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --mem=100G

unset all_proxy   
unset http_proxy       
unset https_proxy 

llamafactory-cli export llama3_lora_sft_merge.yaml
