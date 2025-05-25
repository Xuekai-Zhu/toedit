### ToEdit

This repository hosts the official implementation of the methods introduced in our paper "How to Synthesize Text Data without Model Collapse?" (ICML 2025). 

## Implementation Details

In our paper, we utilize OLMo and LLaMA-Factory for model training. The training data is sourced from:
- [Instruction Pre-Training: Language Models are Supervised Multitask Learners](https://arxiv.org/abs/2406.14491)
- [Princeton-NLP/less_data](https://huggingface.co/datasets/princeton-nlp/less_data)

For detailed data processing and training procedures, please refer to Appendix F in our paper.

We use [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for model evaluation.

## Core Implementation

This repository provides the core implementation of the ToEdit method to facilitate easy migration and adaptation. ToEdit primarily leverages the probability distribution from language models for resampling. Here is the core implementation:

```python
def resampling(prob_dict, num_samples=1, beta=1.5):
    words_candicate = list(prob_dict.keys())
    prob_scores = np.array(list(prob_dict.values()))

    adjusted_probs = softmax(prob_scores / beta) 
    accepted_token = np.random.choice(words_candicate, num_samples, p=adjusted_probs)

    return accepted_token
```

