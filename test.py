from datasets import load_dataset
from datasets import load_from_disk
from tqdm import tqdm
import transformers
from decode_tokens import load_text

all_tokens = 1059594039
print(f"Total tokens: {all_tokens:,} tokens")  # 输出为带逗号分隔符的形式
