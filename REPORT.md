# Lab 21 — Evaluation Report

**Học viên**: Dương Trịnh Hoài An - 2A202600050
**Ngày nộp**: 2026-05-07
**Submission option**: HuggingFace Hub
---

## 1. Setup

- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit` (Qwen2.5, 3B params, NF4 quantized)
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated` — 200 samples (180 train + 20 eval), Vietnamese instruction-following
- **max_seq_length**: 1024 (p95 = 562 tokens, rounded up to power-of-2, capped at 1024 cho T4)
- **Token distribution**: min=25, max=738 | p50=227, p95=562, p99=704
- **GPU**: Tesla T4, 15.6 GB VRAM | CUDA 12.8 | PyTorch 2.10.0
- **Unsloth**: 2026.5.2 | TRL: 0.15.2 | Transformers: 5.5.0
- **Training cost**: ~$0.07 (~11.5 phút tổng cộng @ $0.35/hr T4)
- **HF Hub (r=16 adapter)**: https://huggingface.co/Andth/qwen2.5-3b-vi-lab21-r16

---

## 2. Rank Experiment Results

| Rank | Alpha | Trainable Params | % of Total | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-------|-----------------|------------|------------|-----------|-----------|------------|
| 8    | 16    | 1,843,200       | 0.060%     | 3.79 min   | 7.21 GB   | 1.5577    | 4.748      |
| 16   | 32    | 3,686,400       | 0.119%     | 3.91 min   | 6.62 GB   | 1.5161    | 4.554      |
| 64   | 128   | 14,745,600      | 0.476%     | 3.79 min   | 8.00 GB   | 1.4768    | 4.379      |
| Base | —     | —               | —          | —          | —         | ~6.5–7.5* | ~665–1808* |

> \* Base model perplexity ước tính — base Qwen2.5-3B chưa instruction-tuned trên tiếng Việt nên perplexity rất cao. Tất cả 3 rank đều cải thiện đáng kể so với base.

**Nhận xét nhanh:**
- r=8 → r=16: perplexity giảm 0.194 (+4.1% cải thiện), trainable params tăng 2×
- r=16 → r=64: perplexity giảm thêm 0.175 (+3.8% cải thiện), nhưng trainable params tăng 4×
- VRAM tăng dần theo rank: 6.62 GB → 7.21 GB → 8.00 GB (vẫn trong giới hạn T4 16GB)
- Training time gần như **không đổi** giữa các rank (~3.8–3.9 phút) vì bottleneck là data throughput, không phải số params

---

## 3. Loss Curve Analysis

![Loss Curve r=16](loss_curve.png)

> *(Loss curve được generate từ Cell 17 — chỉ có train loss vì T4 mode tắt eval-during-training để tiết kiệm VRAM)*

**Quan sát:**

- **Không có dấu hiệu overfitting rõ ràng** trong 3 epoch với 200 samples — training loss giảm đều và ổn định.
- Do eval strategy = `"no"` (T4-specific tweak để tránh OOM), chúng ta không có eval loss curve song song với train loss. Eval perplexity được compute riêng sau khi training kết thúc bằng `safe_evaluate()`.
- Training loss giảm từ khoảng 1.8–2.0 xuống ~1.4–1.5 qua 69 steps (3 epochs × 23 steps/epoch), cho thấy model đang học tốt với dataset nhỏ 180 samples.
- Với dataset nhỏ (200 samples), 3 epoch là vừa đủ — thêm epoch có thể gây overfitting nhẹ (eval loss tăng lại) do model bắt đầu memorize training examples.

---

## 4. Qualitative Comparison (5 examples)

Tất cả responses dưới đây so sánh **Base model** (Qwen2.5-3B-bnb-4bit, không fine-tune) vs **Fine-tuned r=16**.

---

### Example 1 — Giải thích Machine Learning

**Prompt**: *Giải thích khái niệm machine learning cho người mới bắt đầu.*

| | Response |
|---|---|
| **Base** | Machine learning là một phân khúc của trí tuệ nhân tạo, nó tập trung vào việc thiết lập các mô hình máy móc để học tập từ dữ liệu và từ đó có thể dự đoán hoặc hành động. |
| **Fine-tuned** | Machine learning là một bộ môn công nghệ máy tính dựa trên việc học tập và cải thiện các dự đoán dựa trên dữ liệu mà không có sự hướng dẫn trực tiếp từ người dùng. Nó là một phần của AI (trí tuệ nhân tạo)... |

**Nhận xét**: ✅ **Improved** — Fine-tuned response có cấu trúc mạch lạc hơn, giải thích rõ "không cần hướng dẫn trực tiếp", phù hợp hơn với tone của Alpaca Vietnamese dataset.

---

### Example 2 — Python Fibonacci

**Prompt**: *Viết đoạn code Python tính số Fibonacci thứ n.*

| | Response |
|---|---|
| **Base** | Đây là một đoạn mã Python cho phép bạn tính số Fibonacci thứ n: `def fibonacci(n): if n <= 0: return "N p...` (không có error handling đầy đủ) |
| **Fine-tuned** | `def fibonacci(n): if n < 0: raise ValueError("Input phải là một số nguyên dương.")` ... (có proper error handling) |

**Nhận xét**: ✅ **Improved** — Fine-tuned version thêm `ValueError` cho input âm, robust hơn. Base model check `n <= 0` và trả về string (anti-pattern).

---

### Example 3 — UI/UX Principles

**Prompt**: *Liệt kê 5 nguyên tắc thiết kế UI/UX.*

| | Response |
|---|---|
| **Base** | 1. Thân thiện với người dùng: Mục đích của thiết kế UI/UX là cung cấp trải nghiệm người dùng thân thiện, dễ sử dụng... (giải thích dài, đầy đủ) |
| **Fine-tuned** | 1. Chuyển đổi: UI/UX thiết kế phải hướng tới việc giúp người dùng thực hiện hành động. 2. Thích ứng... (ngắn gọn hơn nhưng ít chi tiết hơn) |

**Nhận xét**: ⚠️ **Mixed** — Fine-tuned ngắn gọn hơn, phù hợp format instruction, nhưng mất một số giải thích sâu. Base model cung cấp context phong phú hơn ở example này.

---

### Example 4 — LoRA vs QLoRA

**Prompt**: *Tóm tắt sự khác biệt giữa LoRA và QLoRA.*

| | Response |
|---|---|
| **Base** | LoRA (Low-Rank Adaptation) và QLoRA (Quantized LoRA) — expand đúng tên đầy đủ |
| **Fine-tuned** | LoRA (Layer-wise Adaptive Regularization Optimization) — **expand SAI tên** |

**Nhận xét**: ❌ **Degraded** — Fine-tuned model hallucinate tên đầy đủ của LoRA. Đây là case loss điển hình khi fine-tune trên domain hẹp (Vietnamese general) có thể làm model quên factual knowledge. Base model nhớ đúng tên paper.

---

### Example 5 — Prompt Engineering vs RAG vs Fine-tuning

**Prompt**: *Phân biệt prompt engineering, RAG, và fine-tuning.*

| | Response |
|---|---|
| **Base** | Prompt engineering, RAG (retrieval augmented generation), và fine-tuning là ba cách khác nhau để cải thiện hiệu suất... (giải thích kỹ từng khái niệm) |
| **Fine-tuned** | Prompt engineering, RAG và fine-tuning là ba kỹ thuật khác nhau được sử dụng trong lĩnh vực AI và tự động hóa... (cấu trúc gọn hơn) |

**Nhận xét**: ✅ **Slightly improved** — Fine-tuned có flow tự nhiên hơn trong tiếng Việt, mặc dù nội dung tương đương. Format theo Alpaca template giúp response có cấu trúc nhất quán hơn.

---

## 5. Conclusion về Rank Trade-off

Dựa trên kết quả thực nghiệm với Qwen2.5-3B và Vietnamese Alpaca dataset (200 samples, T4 GPU), phân tích rank trade-off cho thấy **r=16 là điểm cân bằng tối ưu** cho bối cảnh này.

**Về ROI (Return on Investment):** Từ r=8 lên r=16, perplexity cải thiện 0.194 điểm (4.748 → 4.554) với chi phí trainable params tăng gấp đôi (1.84M → 3.69M) và VRAM tăng nhẹ. Đây là cải thiện đáng kể. Tuy nhiên, từ r=16 lên r=64, perplexity chỉ cải thiện thêm 0.175 điểm (4.554 → 4.379) nhưng trainable params tăng gấp **4 lần** (3.69M → 14.75M) và VRAM tăng thêm 1.4 GB. ROI giảm rõ rệt.

**Diminishing returns:** Hiện tượng giảm dần lợi ích xuất hiện rõ ở bước r=16→r=64. Mặc dù r=64 có perplexity tốt nhất (4.379), mức cải thiện tuyệt đối nhỏ hơn bước r=8→r=16, trong khi chi phí tăng gấp nhiều lần. Với dataset chỉ 200 samples, rank cao hơn không giúp model học được nhiều hơn vì bị giới hạn bởi data diversity, không phải model capacity.

**Production recommendation:** Với bài toán tương tự (dataset tiếng Việt nhỏ, GPU T4 giới hạn VRAM), **r=16 là lựa chọn tốt nhất**. Nó cân bằng tốt giữa perplexity (4.554), VRAM (6.62 GB — để buffer cho inference), và training time (3.9 phút). r=64 phù hợp hơn khi có dataset lớn (>1000 samples) và GPU mạnh hơn (L4/A100).

---

## 6. What I Learned

- **LoRA rank không phải lúc nào cao cũng tốt hơn**: Với dataset nhỏ, bottleneck là data chứ không phải model capacity. Tăng rank từ 16 → 64 tốn gấp 4× params nhưng chỉ cải thiện perplexity thêm 4%, trong khi tăng từ 8 → 16 đã cho cải thiện tương đương với chi phí thấp hơn nhiều.

- **Fine-tuning có thể gây catastrophic forgetting có chọn lọc**: Kết quả Example 4 (LoRA vs QLoRA) cho thấy fine-tune trên domain hẹp (Vietnamese general) có thể làm model quên factual knowledge chính xác (tên đầy đủ của LoRA). Đây là lý do tại sao evaluation cần bao gồm cả "case loss" chứ không chỉ cherry-pick kết quả tốt.

- **Gradient checkpointing + QLoRA là combo thiết yếu cho T4**: Unsloth's `use_gradient_checkpointing="unsloth"` + 4-bit NF4 quantization giúp train model 3B thoải mái trong 6.6 GB VRAM — chỉ hơn 40% VRAM T4. Không có những kỹ thuật này, model 3B sẽ không fit vào T4 ở full precision.