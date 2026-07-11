# Stage 3 — Emotion & Style Injection

This document describes the joint speaker-emotion conditioning phase, where the model was trained on balanced speaker and emotional style tags to enable expressive speech synthesis.

---

## Objective
The objective of Stage 3 is to inject emotional variations (prosody, intonation, speech rate, and expressive tones) into the synthesized speech while maintaining the speaker profiles established in Stage 2.

## Problem
In Stage 2, the model learned consistent speaker profiles (`prerna` and `prakhar`), but the generated speech remained **monotone**:
1. The model spoke in a flat, neutral rhythm.
2. Subjective reading styles (like expressing surprise, anger, or sadness) were completely absent.
3. Simply training on mixed emotional datasets without explicit tags leads to "style blurring," where the model collapses to a neutral mean.

## Hypothesis
Conditioning the transformer on a joint speaker-style template:
$$\text{Prompt} = \text{"[speaker= speaker_name] [tone= tone_name] \"[text]\""}$$
will allow the self-attention mechanism to learn joint representations. At inference, the model will align the generative audio tokens with both the speaker's vocal profile and the designated emotional style.

---

## Dataset & Preprocessing
* **Size**: A well-balanced dataset of approximately **5,000 samples**.
* **Emotions**: Mapped across **7 emotions** (including anger, surprise, neutral, happy, sad, fear, disgust).
* **Sampling Strategy**: To prevent the model from collapsing to a flat "neutral mean" (catastrophic forgetting of prosody), stratified sampling with replacement was applied to minority emotional styles to hit the exact target proportions.
* **Speaker Balance**: Equal distribution between `prerna` (Female) and `prakhar` (Male).
* **Preprocessing**: Standard audio quality filters (VAD cropping, 2–15s duration, SNR $\ge$ 8.0 dB, clipping fraction $\le$ 0.01, loudness normalized to -23.0 LUFS).

## Method
Supervised Fine-Tuning (SFT) using the checkpoint from Stage 2 as the base. The prompt template was adjusted to feed joint conditioning metadata:
* **Prompt Format**:
  `<|SPEECH_GENERATION_START|> [speaker= speaker_name] [tone= tone_name] "[normalized_text]"`
  `Assistant: <|SPEECH_GENERATION_START|>`
* **Example**:
  `<|SPEECH_GENERATION_START|> [speaker= prakhar] [tone= surprise] "क्या सच में ऐसा हुआ?"`
  `Assistant: <|SPEECH_GENERATION_START|>[audio tokens]`

---

## Training Configuration
* **Base Model Checkpoint**: Stage 2 output model (after speaker-conditioned SFT).
* **LoRA Parameters**: $r = 64$, $\alpha = 64$.
* **Optimizer**: `adamw_8bit`
* **Learning Rate**: $2 \times 10^{-4}$ (linear scheduler).
* **Gradient Clipping**: `max_grad_norm = 0.3`
* **Precision**: `fp16 = True`

## Evaluation
Tested on the 500-sample Hindi benchmark test set, transcribing generated waveforms with Whisper-large-v3 and computing WER and CER via `jiwer`.

---

## Results
The addition of emotion conditioning not only introduced expressive prosody, but further refined pronunciation stability:

| Metric | Stage 2 (Speaker ID) | Stage 3 (Emotion) | Cumulative Improvement (vs Base) |
| :--- | :---: | :---: | :---: |
| **Avg WER (Before Norm)** | 0.4096 | **0.3953** | +57.3% |
| **Avg WER (After Norm)** | 0.3304 | **0.3093** | **+63.7%** |
| **Avg CER (Before Norm)** | 0.2209 | **0.2054** | +77.4% |
| **Avg CER (After Norm)** | 0.1707 | **0.1544** | **+81.5%** |

### Key Takeaway
The transition from Stage 2 to Stage 3 pushed the Word Error Rate down an additional **~2.1%** and the Character Error Rate down another **~1.6%**. Emotion and style tags do not degrade pronunciation accuracy; instead, they act as a regularizer, helping the model learn stronger temporal alignments.

---

## Limitations
* **Style Intensity Boundaries**: Since SFT relies entirely on static, pre-labeled text-audio pairs, the model cannot adjust the *intensity* of an emotion dynamically.
* **Mode Collapse**: Standard SFT cross-entropy optimization tends to penalize out-of-distribution prosodic variations, leading the model to drift back to neutral speech styles under ambiguous prompts.

## Decision
Adopt reinforcement learning to optimize expressiveness dynamically, enabling the model to learn emotional variations directly from a reward signal.

## Next Stage
Design and implement **Stage 4: Group Relative Policy Optimization (GRPO)** for post-SFT reinforcement learning.
