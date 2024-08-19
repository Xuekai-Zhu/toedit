from datasets import load_dataset
from datasets import load_from_disk
from tqdm import tqdm

from decode_tokens import load_text

data = load_text("probability/openwebmath/shard_1_of_8/0_Qwen2-0.5B-Instruct_0.jsonl")
print(len(data))