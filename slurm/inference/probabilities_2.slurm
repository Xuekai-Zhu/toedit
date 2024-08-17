#!/bin/bash
#SBATCH --account=xkzhu
#SBATCH --job-name=token_logits
#SBATCH --partition=RTX4090  
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # 若多卡多进程，请调整此参数
#SBATCH --cpus-per-task=10  # 每个进程的CPU数量
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --qos=high

python inference_vllm.py \
    --model_name_or_path /data1/xkzhu/pre_trained_model/Qwen/Qwen2-0.5B-Instruct \
    --tensor_parallel_size 1 \
    --num_shards 2 \
    --shard_index 2 \
    --gpu_memory_utilization 0.7 \
    --test_file data/instruction_biomed \
    --output_dir probability/biomed_64/