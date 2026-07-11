# Stage 2 — Speaker Identity Fine-Tuning

This document details the speaker-conditioned training phase designed to establish consistent, distinct voice profiles for male and female speakers.

---

## Objective
The objective of Stage 2 is to inject stable and consistent speaker identities into the model, resolving the issue of voice characteristics drifting or shifting randomly during speech generation.

## Problem
Although Stage 1 successfully established Hindi pronunciation alignments, it lacked speaker conditioning. As a result, the model suffered from **voice drift**:
1. The synthesized voice could randomly change gender, pitch, or accent midway through a sentence or across different prompts.
2. The model could not be instructed to generate a specific, reliable voice profile (e.g., a consistent customer assistant voice).

## Hypothesis
Prepending a distinct speaker name prefix to the text prompt (e.g., `SpeakerName: text`) will force the model's self-attention layers to associate specific acoustic properties (pitch range, vocal timbre, speed) with the speaker token. Causal prediction will then condition all subsequent audio tokens on this speaker identity, stabilizing the output voice profile.

---

## Dataset
The training dataset was restricted to the **Rasa Hindi dataset**, which includes explicit gender metadata:
* **Male Voice**: Mapped and tagged with the speaker identity label `prakhar`.
* **Female Voice**: Mapped and tagged with the speaker identity label `prerna`.
* **Preprocessing**: Applied the same filters as Stage 1 (audio segment duration between 2 and 15 seconds, SNR $\ge$ 8.0 dB, clipping fraction $\le$ 0.01, and loudness normalized to -23.0 LUFS).

## Method
Supervised Fine-Tuning (SFT) using the checkpoint from Stage 1 as the base. The text prompts are modified during sequence construction to include the speaker prefix:
$$\text{Formatted Prompt} = \text{SpeakerName} + \text{": "} + \text{text}$$
Example:
`prakhar: हालांकि चिकित्सकों ने जेसीओ की मौत हार्ट अटैक से हुई बताया है`

During training, the standard next-token prediction loss forces the model to align the prefix token (`prakhar` or `prerna`) with the speaker's vocal properties.

---

## Training Configuration
* **Base Model Checkpoint**: Stage 1 output model (after stabilized 4-epoch SFT).
* **LoRA Parameters**: $r = 64$, $\alpha = 64$.
* **Optimizer**: `adamw_8bit`
* **Learning Rate**: $2 \times 10^{-4}$ (linear scheduler).
* **Gradient Clipping**: `max_grad_norm = 0.3`
* **Precision**: `fp16 = True`

## Evaluation
Tested on the 500-sample Hindi benchmark test set, transcribing generated waveforms with Whisper-large-v3 and computing WER and CER via `jiwer`.

---

## Results
Introducing speaker conditioning yielded a massive drop in transcription errors:

| Metric | Stage 1 (Pronunciation) | Stage 2 (Speaker ID) | Relative Improvement (vs Base) |
| :--- | :---: | :---: | :---: |
| **Avg WER (Before Norm)** | 0.7104 | **0.4096** | +55.8% |
| **Avg WER (After Norm)** | 0.6805 | **0.3304** | **+61.2%** |
| **Avg CER (Before Norm)** | 0.4965 | **0.2209** | +75.7% |
| **Avg CER (After Norm)** | 0.4790 | **0.1707** | **+79.5%** |

### Key Takeaway
Constraining the model's vocal search space to two distinct profiles (`prerna` and `prakhar`) significantly stabilized the self-attention alignment maps. The Word Error Rate dropped by **over 51% relatively** (from `0.6805` down to `0.3304`), indicating that speaker consistency directly improves pronunciation stability.

---

## Limitations
* **Monotone Speech**: The model speaks clearly and consistently in the target voice, but the speech is flat and robotic, lacking natural prosodic variation, rhythm, and emotional expression.

## Decision
Proceed to Stage 3 by introducing joint conditioning on speaker identity and emotional styles to enrich speech expressiveness.

## Next Stage
Configure and train **Stage 3 (Emotion & Tone Injection)**.
