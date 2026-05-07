# Links — Lab 21 Submission

## HuggingFace Hub

| Adapter | Link |
|---------|------|
| r=16 (best rank) | https://huggingface.co/Andth/qwen2.5-3b-vi-lab21-r16 |

## Model Card

- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated` (200 samples)
- **Method**: QLoRA 4-bit + LoRA r=16, alpha=32, target: q_proj + v_proj
- **Training**: 3 epochs, lr=2e-4, cosine schedule, T4 GPU