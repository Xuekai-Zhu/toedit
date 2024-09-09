#!/bin/bash
#SBATCH -J inference
#SBATCH -p HGX
#SBATCH --qos=lv0b
#SBATCH --time=10:00:00
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --gres=gpu:1

module load cuda11.8/toolkit/11.8.0
module load cudnn8.6-cuda11.8/8.6.0.163  

source /home/zhuxuekai/miniconda3/bin/activate py39

python inference_vllm_rjs_online.py \
    --model_name_or_path /scratch2/nlp/plm/Meta-Llama-3-8B-Instruct \
    --tensor_parallel_size 1 \
    --num_shards 8 \
    --shard_index 2 \
    --gpu_memory_utilization 0.8 \
    --test_file /home/zhuxuekai/scratch2_nlp/scaling_down_data/data/open-web-math-1B \
    --output_dir probability/open-web-math—1B-up_revise_Llama-3-8B-Instruct/ \
    --n_of_candicant 8 \
    --batch_size 1000 \
    --strategy up_revise