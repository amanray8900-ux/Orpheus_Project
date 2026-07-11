# Stage 1 — Pronunciation Fine-Tuning

This document describes the fine-tuning of the model for Devanagari script pronunciation, the analysis of the initial 5-epoch training failure, and the successful 4-epoch retraining on a stabilized dataset.

---

## Objective
The objective of Stage 1 is to teach the model Devanagari character-phoneme mappings, stabilize word alignment, and establish a baseline pronunciation capability in Hindi and Hinglish.

---

## Problem: The 5-Epoch Failure
The first training run was conducted on 42,851 samples (comprising 50% Rasa, 50% Kathbath, 20% Indic Voices, and the complete Hinglish dataset) for 5 epochs. This run crashed and failed due to three concurrent issues:

1. **The "Epoch Reset" Learning Rate Bug**:
   * *Mechanism*: The training was resumed from an active checkpoint, but the `num_train_epochs` parameter was modified (changed from 1 to 3).
   * *Impact*: When `num_train_epochs` or `max_steps` is adjusted during a resume in Hugging Face Trainer, the existing learning rate schedule is discarded. A new warmup cycle was initiated, causing the learning rate to spike from a stable `0.000025` (at step 1000) to `0.000171` (at step 1010). This sudden gradient shock obliterated the previously learned alignments.
2. **Exploding Gradients and NaN Crash**:
   * *Mechanism*: A second resume adjustment was made at step 3500 (changing epochs to 4) without enabling gradient clipping (`max_grad_norm` was undefined).
   * *Impact*: Learning rate shocks caused gradients to grow exponentially. By step 4600, values exceeded numerical limits, resulting in a **NaN loss crash**.
3. **Out-of-Memory (OOM) Failures**:
   * *Mechanism*: Audio durations in the dataset were unconstrained, containing segments longer than 15 seconds.
   * *Impact*: Long audio sequences caused sequence lengths to exceed the 2048 context limit, triggering VRAM overflows on the GPU.

---

## Hypothesis
Training can be stabilized and OOMs prevented by:
1. **Strict Data Filtering**: Limiting audio segment duration to **2–15 seconds**, enforcing a minimum Signal-to-Noise Ratio (SNR) of **8.0 dB**, and a maximum clipping fraction of **0.01 (1%)**.
2. **Trainer Stability Guards**: Resuming only with strict parameter overrides: locking `num_train_epochs` during resumes, enabling half-precision training (`fp16 = True`), and enforcing gradient clipping (`max_grad_norm = 0.3`).

---

## Dataset & Preprocessing
The model was retrained on a reduced, high-quality subset of **31,394 samples (approx. 52 hours of data)**:
* **Composition**: 100% Hinglish (Ayushi Agarwal dataset), 30% Kathbath, and 50% Rasa dataset.
* **Audio Filters**:
  * Duration range: $2.0 \le t \le 15.0$ seconds.
  * Voice Activity Detection (VAD): Applied `silero-vad` to crop silent leading/trailing frames.
  * Quality thresholds: SNR $\ge$ 8.0 dB, clipping fraction $\le$ 0.01.
  * Loudness: Normalized to **-23.0 LUFS** via `pyloudnorm`.
* **Text Normalization**:
  * Executed the custom `HindiNormalizer` class to clean Devanagari text.
  * Expanded numbers (e.g., "5" $\rightarrow$ "पाँच"), percentages, currencies, dates, times, and math symbols.
  * Unified character representations using strict NFC normalization, handling Urdu nuktas (e.g., standardizing ड़ and ढ़).
  * Removed stray punctuation and double whitespaces.

---

## Method
Supervised Fine-Tuning (SFT) using Unsloth's `FastLanguageModel` to train the LoRA adapters on the target attention and MLP projections.

---

## Training Configuration
* **Backbone Model**: `unsloth/orpheus-3b-0.1-ft`
* **Trainable Parameters**: LoRA adapters ($r = 64$, $\alpha = 64$, dropout = 0) applied to attention projections.
* **Epochs**: 4
* **Optimizer**: `adamw_8bit`
* **Learning Rate**: $2 \times 10^{-4}$ (linear scheduler, warmup = 5 steps)
* **Gradient Clipping**: `max_grad_norm = 0.3` (critical fix to prevent exploding gradients)
* **Batch Size**: 1 (gradient accumulation = 4, effective batch size = 4)
* **Precision**: `fp16 = True` (or `bfloat16 = True` depending on GPU support)

---

## Evaluation
Evaluated on the 500-sample Hindi benchmark test set, using Whisper-large-v3 to transcribe outputs and calculating Word Error Rate (WER) and Character Error Rate (CER) via the `jiwer` library.

---

## Results
The model stabilized during training, successfully avoiding NaN crashes:
* **Average WER (Before Norm)**: `0.7104`
* **Average WER (After Norm)**: `0.6805` (**+20.1% improvement** vs. Base Model `0.8524`)
* **Average CER (Before Norm)**: `0.4965`
* **Average CER (After Norm)**: `0.4790` (**+42.6% improvement** vs. Base Model `0.8351`)

---

## Limitations
1. **Generic / Drifting Speaker Identity**: The model generates audible Hindi speech, but the voice characteristics change randomly between prompts, or lack a clear, human-like speaker profile.
2. **Monotone Prosody**: The speech lacks emotional intonation or expressive variations.

## Decision
Proceed to Stage 2 by conditioning the training dataset on explicit speaker identities to stabilize voice profile generation.

## Next Stage
Configure and train **Stage 2 (Speaker Identity Fine-Tuning)**.
