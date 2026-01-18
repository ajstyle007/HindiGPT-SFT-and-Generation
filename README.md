## HindiGPT — SFT, Generation & Frontend

A complete Supervised Fine-Tuning (SFT), controlled text generation, and Gradio-based frontend pipeline built on top of a custom GPT architecture trained from scratch for Hindi language.

### This repository covers:

- ✅ Supervised Fine-Tuning (SFT) on Hindi Q&A data
- ✅ Advanced text generation (Top-K, Top-P, Temperature, Repetition Penalty)
- ✅ Interactive Gradio web application for inference

### 1. Supervised Fine-Tuning (SFT)
This repository performs Supervised Fine-Tuning (SFT) on a pretrained Hindi GPT model to enable instruction-following and question–answering behavior.

- The model is fine-tuned using Alpaca-style Hindi Q&A pairs
- Training uses a decoder-only Transformer with causal language modeling loss

The prompt format strictly follows:
```
### प्रश्न:
<user question>

### उत्तर:
<expected answer>
```

#### Dataset Details

- Source: Hugging Face Alpaca-style dataset
- Language: Hindi
- Data Type: GPT-4 generated instruction-following data

- Cleaning:
 - Removed noisy / empty samples
 - Normalized Unicode & whitespace
 - Ensured consistent Q&A formatting

Final dataset size: 36,353 high-quality Hindi instruction samples

🔹 Training Highlights

- Optimizer: AdamW
- Scheduler: Cosine Annealing
- Epochs: 5
- Stable convergence with decreasing loss & perplexity
- Periodic checkpointing every 10k steps

The result is a strong Hindi instruction-tuned model capable of answering factual, conceptual, and creative questions.

### 2. Advanced Text Generation

A custom generation pipeline is implemented from scratch to ensure controlled, high-quality Hindi text generation.

- Supported Techniques
- Top-K Sampling – limits token choices to most probable K tokens
- Top-P (Nucleus Sampling) – dynamic probability mass sampling
- Temperature Scaling – controls randomness & creativity
- Repetition Penalty – reduces looping & token repetition
- EOS-aware stopping – graceful termination

Default Optimized Parameters

These defaults are tuned for balanced, fluent Hindi output:

| Parameter	| Value |
| ---------- | ------- |
| Temperature	| 0.85 |
| Top-K	| 45 |
| Top-P	| 0.92 | 
| Max Tokens |	300 |
| Repetition Penalty |	1.12 |

Additional safeguards are added to handle:

- NaN / Inf logits
- Probability renormalization
- Stable sampling under low temperature

This generation setup works reliably on CPU and GPU, making it suitable for deployment environments like Hugging Face Spaces.

### 3. Interactive Gradio Frontend ([Live Web App](https://musk12-hindi-gpt-model-built-from-scratch.hf.space/))

An interactive Gradio-based web application is built for real-time inference and experimentation.

<img width="1722" height="807" alt="hindi_gpt_web_app" src="https://github.com/user-attachments/assets/8db918d7-9b6f-473c-823c-3e5b1637104d" />

#### Frontend Features
- Hindi-first UI
- Clean dark theme
- Adjustable generation parameters:
   - Top-K
   - Top-P
   - Temperature
   - Max Tokens

- Example prompts for quick testing
- Clear & Generate actions
- Post-processing to return only the answer text

Usage
```
python app2.py
```

The UI allows both research-style probing and casual user interaction, making the model easy to demo and share.

