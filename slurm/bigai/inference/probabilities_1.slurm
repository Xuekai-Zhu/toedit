#!/bin/bash
#SBATCH -J inference
#SBATCH -p HGX
#SBATCH --qos=lv0b
#SBATCH --time=10:00:00
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --gres=gpu:1

python inference_vllm_rjs.py \
    --model_name_or_path /scratch2/nlp/plm/Qwen2-0.5B-Instruct \
    --tensor_parallel_size 1 \
    --num_shards 2 \
    --shard_index 1 \
    --gpu_memory_utilization 0.8 \
    --test_file data/instruction_biomed \
    --output_dir probability/biomed_64/