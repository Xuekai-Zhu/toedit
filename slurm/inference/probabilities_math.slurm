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

python inference_vllm_rjs_online_for_math.py \
    --model_name_or_path /data1/xkzhu/pre_trained_model/meta-llama/Meta-Llama-3-8B \
    --tensor_parallel_size 1 \
    --num_shards 8 \
    --shard_index 1 \
    --gpu_memory_utilization 0.8 \
    --test_file data/math/open-web-math-1B \
    --output_dir probability/test \
    --n_of_candicant 8 \
    --batch_size 1000 \
    --strategy up_revise