import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = load_dataset(args.dataset_name, split="train", cache_dir=f"{args.dataset_name}/cache/", num_proc=12)
    # dataset = dataset.filter(
    #     lambda x, i: i % args.num_shards == args.shard, with_indices=True, num_proc=12
    # )

    def compute_scores(batch):
        messages = [i + " " + j for i,j in zip(batch["problem"], batch["solution"])]
        inputs = tokenizer(
            messages,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.squeeze(-1).float().cpu().numpy()

        batch["score"] = logits.tolist()
        batch["int_score"] = [int(round(max(0, min(score, 5)))) for score in logits]
        return batch

    dataset = dataset.map(compute_scores, batched=True, batch_size=args.batch_size)
    dataset.save_to_disk(args.output_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name", type=str, default="HuggingFaceFW/fineweb-edu-classifier"
    )
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb")
    # parser.add_argument("--dataset_config", type=str, default="default")
    parser.add_argument(
        "--output_dir", type=str, default="baselines/edu_scores"
    )
    # parser.add_argument("--output_dataset_config", type=str, default="default")
    # parser.add_argument("--text_column", type=str, default="text")
    # parser.add_argument("--shard", type=int, required=True)
    # parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument(
        "--batch_size", type=int, default=32
    )

    args = parser.parse_args()
    main(args)