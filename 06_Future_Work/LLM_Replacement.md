# LLM Backbone Replacement

This document outlines the design and proposed evaluation for replacing the default Llama 3B backbone model with other, more optimized transformer models.

---

## Objective
The objective is to explore alternative transformer backbones (specifically **Gemma-3-270M**, **Qwen3-4B**, and **Gemma-3-4B**) to optimize memory efficiency, inference speed, and token density for Hindi speech generation.

## Problem
The current 3B parameter Llama-based backbone is computationally heavy:
1. **High Memory Overhead**: LoRA training requires ~6.5 GB VRAM, which limits the batch size and makes reinforcement learning (GRPO) difficult to scale on low-memory GPUs.
2. **Tokenizer Efficiency**: The standard Llama tokenizer represents Hindi characters with multiple byte-level tokens, leading to unnecessarily long sequences for short text prompts. This wastes context length and slows down next-token generation.
3. **Inference Latency**: Autoregressive generation of 1200+ audio tokens at 3B parameters can suffer from high latency on edge devices.

## Hypothesis
1. **Gemma-3-270M (Extreme Efficiency)**: Replacing the 3B model with a 270M parameter model will reduce parameter count by over 90%, cutting VRAM requirements to <1 GB and enabling real-time edge/CPU speech generation, without a catastrophic drop in pronunciation stability.
2. **Qwen3-4B / Gemma-3-4B (Vocab Density)**: Multilingual-optimized models like Qwen3-4B have larger, highly efficient vocabularies. This will compress Hindi text sequence length by 30-40%, allowing longer audio generation within the 2048 token limit.

---

## Dataset
Evaluated on:
* The stabilized Stage 1 Pronunciation dataset (31,394 samples).
* The Stage 3 Speaker-Emotion dataset (5,000 samples).

## Method
1. **Model Loading**: Initialize the candidate base model (e.g., `unsloth/Qwen3-4B-unsloth-bnb-4bit` or `google/gemma-3-270m`) using FastLanguageModel.
2. **Vocabulary Extension**: Add the special boundary tokens (SOH, EOT, EOH, SOA, SOS, EOS, EOA) and register the offsets for the 28,672 vocab spaces required by the 7-token SNAC frame ranges.
3. **LoRA Layer Target Mapping**: Map and target the specific query projections for the new architecture (e.g., matching the attention layers in Gemma or Qwen).
4. **SFT Retraining**: Fine-tune the new backbone through SFT Stages 1-3.
5. **A/B Benchmark**: Compare performance metrics against the Llama 3B SFT Stage 3 model.

---

## Training Configuration
* **Candidates**:
  * `unsloth/Qwen3-4B-unsloth-bnb-4bit`
  * `unsloth/Qemma-3-270m` (or equivalent Gemma-3-270M checkpoint)
  * `unsloth/gemma-3-4b-it-unsloth-bnb-4bit`
* **PEFT**: LoRA rank $r=64$, $\alpha=64$.
* **Optimizer**: `adamw_8bit`
* **Learning Rate**: $2 \times 10^{-4}$

---

## Evaluation
The backbones will be compared across four key metrics:
1. **Phonetic Accuracy**: Avg WER and CER on the 500-sample test set (transcribed via Whisper-large-v3).
2. **Inference Speed**: Generation speed measured in audio tokens generated per second.
3. **Memory Footprint**: Peak VRAM usage during training and rollout generation.
4. **Compression Ratio**: Average number of tokens required to represent 100 characters of Hindi text.

---

## Limitations
* **Representational Limits**: A smaller model like Gemma-3-270M may struggle with joint speaker-emotion conditioning, resulting in voice leakage (cross-talk between Prerna and Prakhar profiles).
* **Quantization Noise**: Running in 4-bit (`load_in_4bit = True`) can introduce quantization noise into audio token generation, causing minor background hiss.

## Decision
Prioritize A/B testing of Gemma-3-270M to evaluate minimum VRAM boundaries, and use Qwen3-4B for applications requiring higher pronunciation accuracy.

## Next Stage
Establish evaluation criteria for **Emotion and Speaker consistency** in generated outputs.
