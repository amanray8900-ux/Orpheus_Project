# Evaluation Metrics Summary

This document summarizes the quantitative results achieved across all stages of fine-tuning, comparing Word Error Rate (WER) and Character Error Rate (CER) against the baseline.

---

## Objective
The objective is to consolidate and compare the transcription error rates (WER and CER) across all stages of model development to track accuracy improvements and quantify model progress.

## Problem
Comparing models across stages without a standardized metric suite and normalizer leads to inconsistent evaluation. Minor orthographic variations in Devanagari (like optional nuktas, spacing, or digit format differences) can artificially inflate error rates, masking actual pronunciation stability gains.

## Hypothesis
Computing WER and CER both *before* and *after* strict Devanagari text normalization will isolate phonetic spelling differences from core pronunciation errors. Standardizing evaluations on a static 500-sample test set will clearly demonstrate progressive error reduction across training stages.

---

## Dataset
Evaluated on the benchmark dataset of **500 samples** spanning Phases 1-3 from `dataset_report.md` (representing basic pronunciation, linguistic variations, and real-world news/books/dialogue data).

## Method
* Waveforms are synthesized using each checkpoint.
* Audio files are transcribed using OpenAI `whisper-large-v3`.
* Metrics are calculated using `jiwer`.
* Normalization is done via the `HindiNormalizer` preprocessing script.

---

## Training Configuration
Not applicable (post-training evaluation summary).

---

## Results: Metrics Comparison Table

The tabulated evaluation metrics across all model checkpoints are presented below:

| Model Version | Avg WER (Before Norm) | Avg WER (After Norm) | Avg CER (Before Norm) | Avg CER (After Norm) | WER Improvement (vs Base) | CER Improvement (vs Base) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base Model** | `0.9261` | `0.8524` | `0.9085` | `0.8351` | *Baseline* | *Baseline* |
| **Stage 1 (Pron.)** | `0.7104` | `0.6805` | `0.4965` | `0.4790` | **+20.1%** | **+42.6%** |
| **Stage 2 (Spk ID)** | `0.4096` | `0.3304` | `0.2209` | `0.1707` | **+61.2%** | **+79.5%** |
| **Stage 3 (Emotion)** | `0.3953` | `0.3093` | `0.2054` | `0.1544` | **+63.7%** | **+81.5%** |

### Key Takeaways

1. **Uninterrupted Progress**: Every single stage of fine-tuning yielded a strictly better model. The transcription errors dropped consistently at each checkpoint without regression.
2. **Massive Overall Gains**: By Stage 3, the Word Error Rate (WER) was reduced by **almost 64%** and the Character Error Rate (CER) by **over 81%** compared to the zero-shot base model.
3. **Stage 3 Polish**: The transition from Stage 2 to Stage 3 pushed the Word Error Rate down an additional **~2.1%** and the Character Error Rate down another **~1.6%**. Stage 3 successfully introduced emotional control and continued to refine the model's core pronunciation stability.

---

## Limitations
* **ASR Dependencies**: Transcription errors in Whisper-large-v3 (such as spelling name tokens incorrectly) are counted as model synthesis errors.
* **Lack of Subjective Metrics**: Edit distance error rates do not capture sound quality aspects like background noise, clipping, or natural phrasing.

## Decision
Accept the Stage 3 checkpoint as the standard SFT production candidate and establish it as the reference model ($\pi_{ref}$) for future RL/GRPO training.

## Next Stage
Compile instructions and guidelines for curating **Audio Examples** to verify model sound quality.
