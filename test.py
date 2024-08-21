from datasets import load_dataset
from datasets import load_from_disk
from tqdm import tqdm
import transformers
from decode_tokens import load_text

tokenizer = transformers.AutoTokenizer.from_pretrained("pre_trained_model/Qwen2-0.5B-Instruct")
batch_prompts = ["i am you i am you i am you i am you i am you", "i am"]
prompt_token_ids = tokenizer(batch_prompts, truncation=True, max_length=4)["input_ids"]
print(prompt_token_ids)