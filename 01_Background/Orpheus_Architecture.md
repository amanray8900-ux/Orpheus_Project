# Orpheus Architecture & Causal Flow

This document details the causal Text-to-Speech (TTS) flow of the Orpheus model, mapping how inputs progress from text strings to synthesized waveforms through token generation and audio codec reconstruction.

---

## Objective
The objective is to repurpose a large, pre-trained text language model (specifically, a ~3B parameter Transformer) as a causal speech generator by framing TTS as a next-token prediction task.

## Problem
Traditional TTS pipelines involve separate acoustic modeling (e.g., text-to-mel-spectrogram regression) and neural vocoding stages. These pipelines suffer from:
1. Optimization bottlenecks due to multi-stage dependencies.
2. Robotic or averaged prosody caused by MSE/L1 regression loss over continuous features.
3. Incapacity to leverage the rich contextual understanding of pre-trained LLMs.

## Hypothesis
Casting speech generation as causal sequence modeling over a unified vocabulary—containing both standard text characters and discrete audio tokens—allows the model to generate highly natural speech. The LLM's self-attention mechanism learns to map orthographic context to prosodic contours and phonetic transitions.

## Dataset
Initialized on English datasets like `Etherll/kaira` and adapted to Hindi using Rasa, Kathbath, and Indic Voices.

---

## Method: Causal Text-to-Speech Flow

The end-to-end flow of the Orpheus TTS model operates as a dual-stage pipeline:

```
[ Text Input ] 
       │
       ▼ (Prompt Formatting)
"User: <speaker> <style> बोलिए : '[text]'"
       │
       ▼ (Llama 3.2-3B Tokenization)
Input Sequence: [SOH] Prompt Tokens [EOT] [EOH] [SOA] [SOS]
       │
       ▼ (Stage 1: Autoregressive Next-Token Generation)
Generated Output: Stream of custom audio tokens (<custom_token_N>)
       │
       ▼ (Stage 2: Token-to-Code Offset Translation)
Parsed Codebook Indices: L1, L2, L3 codes
       │
       ▼ (SNAC Decoder & Overlap-Add Windowing)
Output: 24kHz Mono 16-bit WAV bytes
```

### Stage 1: Language Modeling (LLaMA 3B)
1. **Input Preparation**: The user's text prompt is expanded with conditioning markers (e.g. `prakhar: बोलिए : "..."`). Special boundary tokens are appended to transition the transformer state from text prompt reading to speech generation.
2. **Causal Next-Token Prediction**: The model processes the token sequence and predicts the next token in the sequence from a vocabulary expanded by **28,682 custom tokens** (offset to prevent collision with text).
3. **Autoregressive Generation**: The model takes its own predicted audio token, appends it to the context window, and predicts the subsequent token.

### Stage 2: SNAC Audio Decoding
1. **Stream Parsing**: The generated `<custom_token_N>` string IDs are parsed back into integers.
2. **Offset Extraction**: The codebook indices are extracted using the slot position modulo 7 (see [SNAC.md](file:///c:/Users/Admin/Desktop/context/Orpheus_Project/01_Background/SNAC.md) for math).
3. **Reconstruction**: The SNAC decoder takes the compiled codebook arrays and decodes them back to a 24,000 Hz float32 audio waveform.

---

## Training Configuration
* **Optimizations**: LoRA PEFT ($r=64$, $\alpha=64$), gradient checkpointing (`use_gradient_checkpointing = "unsloth"`), and 8-bit AdamW optimizer.
* **Teacher Forcing**: Ground-truth target tokens are fed in at each step during training to stabilize cross-entropy loss computation.

## Evaluation
Causal cross-entropy loss is computed during training. Text loss quickly approaches zero, meaning the model's parameters are updated almost exclusively on audio token predictions.

## Results
The causal flow successfully synthesizes high-quality speech. Reusing LLM semantic features generates natural phrasing and rhythm, reflecting an understanding of punctuation and word meanings.

## Limitations
1. **Exposure Bias**: During generation, the model relies on its own past predictions. If a single audio token is predicted incorrectly, the error can cascade, causing stutters or phonetic hallucinations.
2. **Latency Bottlenecks**: Generating tokens one by one (autoregressively) is computationally bound by memory bandwidth.

## Decision
Utilize the causal dual-stage autoregressive flow, implementing Unsloth native optimizations and vLLM to achieve low latency.

## Next Stage
Configure the **SNAC Audio Codec** to handle waveform compression and reconstruction math.
