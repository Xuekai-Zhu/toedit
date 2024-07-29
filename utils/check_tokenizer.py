from transformers import AutoTokenizer
from olmo import Tokenizer
# Load the tokenizer

# tokenizer = AutoTokenizer.from_pretrained("/data1/xkzhu/pre_trained_model/allenai/OLMo-1B")


tokenizer = Tokenizer.from_file("OLMo/olmo_data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json")
# tokenizer = Tokenizer.from_file("OLMo/olmo_data/tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json")

print(tokenizer)
# tokenizer.pad_token_id = 0
# Check and print special tokens
# print("BOS token:", tokenizer.bos_token, tokenizer.bos_token)
print("EOS token:", tokenizer.eos_token, tokenizer.eos_token_id)
print("PAD token:", tokenizer.pad_token, tokenizer.pad_token_id)
# print("UNK token:", tokenizer.unk_token)
# print("SEP token:", tokenizer.sep_token)
# print("CLS token:", tokenizer.cls_token)
# print("MASK token:", tokenizer.mask_token)

# Check if the tokenizer adds BOS token
# print("Adds BOS token:", tokenizer.add_special_tokens({'bos_token': '[BOS]'}))

# Check if PAD token ID exists and print it
# pad_token_id = tokenizer.pad_token_id
# if pad_token_id is not None:
#     print("PAD token ID exists:", pad_token_id)
# else:
#     print("PAD token ID does not exist")

# Example text to tokenize
text = ["This is a test sentence.", "is a test sentence."]

# Tokenize the text
for i in text:
    tokens = tokenizer.encode(i)
    print("Tokens:", tokens)
    # input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # print("Input IDs:", input_ids)

# # Convert tokens to input IDs
# input_ids = tokenizer.convert_tokens_to_ids(tokens)
# print("Input IDs:", input_ids)

# # Encode the text (convert to input IDs with added special tokens)
# encoded_input = tokenizer(text, return_tensors='pt', padding='longest', truncation=True)
# print("Encoded input:", encoded_input)

# # Decode the input IDs back to text
# decoded_text = tokenizer.decode(encoded_input['input_ids'][0], skip_special_tokens=True)
# print("Decoded text:", decoded_text)