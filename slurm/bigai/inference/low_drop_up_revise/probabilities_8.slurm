#!/bin/bash
#SBATCH -J inference
#SBATCH -p HGX
#SBATCH --qos=lv2
#SBATCH --time=2-00:00:00
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --gres=gpu:1
#SBATCH --account=research  

module load cuda11.8/toolkit/11.8.0
module load cudnn8.6-cuda11.8/8.6.0.163  

source /home/zhuxuekai/miniconda3/bin/activate py39


python inference_vllm_rjs_online.py \
    --model_name_or_path /scratch2/nlp/plm/Qwen2-0.5B-Instruct \
    --tensor_parallel_size 1 \
    --num_shards 8 \
    --shard_index 8 \
    --gpu_memory_utilization 0.8 \
    --test_file data/instruction_biomed \
    --output_dir probability/instruction_biomed—1B-low_drop_up_revise/ \
    --n_of_candicant 64 \
    --batch_size 1000 \
    --strategy low_drop_up_revise