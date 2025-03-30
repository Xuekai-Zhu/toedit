#!/bin/bash

#SBATCH --time=10:00:00
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --gres=gpu:1
#SBATCH --account=xkzhu
#SBATCH --job-name=inference
#SBATCH --partition=A100


python inference_vllm_rjs_online_for_sft_statistic_p_icml_rebuttal.py \
    --model_name_or_path /data1/xkzhu/pre_trained_model/meta-llama/Meta-Llama-3-8B-Instruct \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.8 \
    --test_file data/less-data-llama-revised-2/cot-json-gz \
    --output_dir probability/statistics/ICML_rebuttal \
    --n_of_candicant 8 \
    --batch_size 1000 \
    --strategy up_revise