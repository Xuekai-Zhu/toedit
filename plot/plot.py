import matplotlib.pyplot as plt
import json
from transformers import AutoTokenizer
import math
import os

def load_sample(in_file, sample_line=0):
    with open(in_file, "r") as f:
        data = f.readlines()
    
    return json.loads(data[sample_line])
        
        
def plot_token_probabilities(tokenizer, json_file, segment_size, save_path):
    sample = load_sample(json_file)
    
    logprobs = sample["prompt_logprobs"][1:]
    token_ids = sample["prompt_token_ids"][1:]
    
    assert len(logprobs) == len(token_ids)
    
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    low_prob_threshold = 0.001
    high_prob_threshold = 0.99
    # log_low_prob_threshold = math.log(low_prob_threshold)
    # log_high_prob_threshold = math.log(high_prob_threshold)
    probs = [math.exp(logprob) for logprob in logprobs]
   
    
    for i in range(0, len(tokens), segment_size):
        segment_tokens = tokens[i:i+segment_size]
        segment_probabilities = probs[i:i+segment_size]
        
        plt.figure(figsize=(15, 7))
        # plt.scatter(range(len(segment_tokens)), segment_probabilities, color='blue', label='Log Probabilities')
        
        for idx, (token, prob) in enumerate(zip(segment_tokens, segment_probabilities)):
            if prob < low_prob_threshold:
                plt.scatter(idx, prob, color='red', marker='s', label=f'Low Probability ({low_prob_threshold})' if idx == 0 else "")
            elif prob > high_prob_threshold:
                plt.scatter(idx, prob, color='green', marker='^', label=f'High Probability (>{high_prob_threshold})' if idx == 0 else "")
            else:
                plt.scatter(idx, prob, color='blue', label='Log Probabilities' if idx == 0 else "")
        
        
        # 画出低概率阈值线并增加说明
        plt.axhline(y=low_prob_threshold, color='red', linestyle='--')
        plt.text(len(segment_tokens) - 1, low_prob_threshold - 0.005, f'{low_prob_threshold} Probability Threshold', color='red', va='top', ha='right')

        # 画出高概率阈值线并增加说明
        plt.axhline(y=high_prob_threshold, color='green', linestyle='--')
        plt.text(len(segment_tokens) - 1, high_prob_threshold - 0.005, f'{high_prob_threshold} Probability Threshold', color='green', va='top', ha='right')
        
        # 标记低概率和高概率 token 并倾斜显示
        xtick_labels = []
        xtick_colors = []
        for token, prob in zip(segment_tokens, segment_probabilities):
            if prob < low_prob_threshold:
                xtick_labels.append(token)
                xtick_colors.append('red')
            elif prob > high_prob_threshold:
                xtick_labels.append(token)
                xtick_colors.append('green')
            else:
                xtick_labels.append(token)
                xtick_colors.append('black')
        
        plt.xticks(range(len(segment_tokens)), xtick_labels, rotation=45, ha='right')
        
        # 设置 x 轴标签颜色
        ax = plt.gca()
        for tick_label, color in zip(ax.get_xticklabels(), xtick_colors):
            tick_label.set_color(color)
        
        plt.xlabel('Tokens', fontsize=23)
        plt.ylabel('Probabilities', fontsize=23)
        plt.title(f'Probabilities of Tokens (Segment {i//segment_size+1})')
        # plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        out_file = os.path.join(save_path, f'segment_{i//segment_size + 1}.png')
        plt.savefig(out_file, dpi=500)
        plt.close()

if __name__ == '__main__':
    save_path = "plot/biomed/Qwen2-0.5B-Instruct_0/0.001and0.99"
    os.makedirs(save_path, exist_ok=True)
    in_file = "probability/biomed/shard_1_of_2/0_Qwen2-0.5B-Instruct_0.jsonl"
    segment_size = 50
    
    tokenizer = AutoTokenizer.from_pretrained("/data1/xkzhu/pre_trained_model/Qwen/Qwen2-0.5B")
    plot_token_probabilities(tokenizer, in_file, segment_size, save_path)
