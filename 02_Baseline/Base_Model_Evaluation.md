# Base Model Evaluation (Baseline Benchmark)

This document describes the objective, methodology, and results of evaluating the zero-shot capabilities of the base Orpheus TTS model on Hindi speech synthesis.

---

## Objective
The objective is to establish a solid performance baseline (benchmark) by evaluating the zero-shot Hindi speech generation capability of the un-tuned base model (`unsloth/orpheus-3b-0.1-ft`).

## Problem
The base Orpheus model was primarily fine-tuned on English speech corpora. Its ability to align Devanagari text prompts with Devanagari/Hindi phonemes, handle Devanagari characters, and synthesize Hindi waveforms is untested, and is expected to be extremely poor without target-language supervised fine-tuning.

## Hypothesis
Causal zero-shot inference on Devanagari prompts will result in severe alignment failures, phonetic hallucinations, high accents, or complete unintelligibility, manifesting as Word Error Rate (WER) and Character Error Rate (CER) scores exceeding 80%.

## Dataset
The benchmark is run on a compiled dataset of **500 evaluation samples** taken from Phases 1, 2, and 3 of the project manifest (as described in [dataset_report.md](file:///c:/Users/Admin/Desktop/context/dataset_report.md)):

1. **Phase 1: Basic Pronunciation (50 Samples)**
   * Sources: Rasa_AI4Bharat (20), Shekharmeena_TTS (20), IndicVoices (10).
   * Text Statistics: Average 14.5 words, 69.4 characters per sample.
2. **Phase 2: Linguistic Variation (200 Samples)**
   * Sources: LLM_Generated (80), Rasa_AI4Bharat (40), IndicVoices (40), Shekharmeena_TTS (40).
   * Text Statistics: Average 21.2 words, 102.1 characters per sample.
3. **Phase 3: Real World Content (250 Samples)**
   * Sources: Rasa_AI4Bharat (50), IndicVoices (50), Shekharmeena_TTS (50), Hindi_Books_Audio (50), Hindi_Story_News (50).
   * Text Statistics: Average 20.8 words, 101.1 characters per sample.

---

## Method
1. **Audio Generation**: Text prompts from the 500 benchmark samples are fed into the base model. Audio tokens are generated autoregressively and decoded to 24kHz waveforms via SNAC.
2. **Automatic Speech Recognition (ASR)**: The generated audio files are transcribed back into text using OpenAI's **Whisper-large-v3** model configured for Hindi.
3. **Error Analysis**: Transcriptions are compared to the ground-truth text targets. Word Error Rate (WER) and Character Error Rate (CER) are calculated using the `jiwer` library.
4. **Normalization Comparison**: Metrics are computed in two configurations:
   * **Before Norm**: Raw transcript vs. raw ground-truth.
   * **After Norm**: Applying a strict Devanagari normalizer (`HindiNormalizer` from the preprocessing pipeline) to both text streams to remove minor punctuation and standardize numbers/digits.

---

## Training Configuration
* **Training Stage**: None (Zero-shot baseline evaluation).
* **Generation Parameters**:
  * Temperature: `0.6`
  * Top-p: `0.95`
  * Repetition Penalty: `1.1`
  * Max new tokens: `1200`

---

## Evaluation
Calculated using the `jiwer` library:
$$\text{WER} = \frac{S + D + I}{N}$$
Where $S$ is substitutions, $D$ is deletions, $I$ is insertions, and $N$ is the number of words in the ground-truth text.

---

## Results
The baseline evaluation metrics are compiled in the table below:

| Metric | Before Normalization | After Normalization |
| :--- | :---: | :---: |
| **Average Word Error Rate (Avg WER)** | `0.9261` | `0.8524` |
| **Average Character Error Rate (Avg CER)** | `0.9085` | `0.8351` |

## Limitations & Failure Analysis (Sentence-Specific Patterns)

Based on the [Compiled Evaluation Report](file:///c:/Users/Admin/Desktop/context/compiled_sample_report.md), the baseline model struggles with specific linguistic and structural sentence patterns:

1. **Long and Complex Sentences**: 
   * In Phase 3 (dominated by long, complex sentences), the baseline model suffers from extreme attention drift. Causal attention maps degrade over long contexts, causing the generated audio to cut off, drift into static noise, or hallucinate repetitive syllables. This results in a high average WER of `1.3052` for Phase 3 normalized audio.
2. **Domain-Specific Terminology and Loanwords**:
   * The model shows high error rates in domains like **Storytelling** (WER `1.5867`) and **Government** (WER `0.8185`). Sentences containing specialized political terminology, classical literary vocabulary, or English/Hinglish technical loanwords fail to align properly, leading to phonetic collapses where the model outputs garbled speech.
3. **Numerals and Mathematical Symbols**:
   * Sentences containing raw digits or abbreviations (e.g., years, dates, percentages) cause major failures. Without text normalization, the baseline model attempts to pronounce digits as individual characters (often in English) rather than standard Hindi word equivalents, causing high WER mismatch in ASR evaluation.
4. **Devanagari Conjuncts and Nuktas**:
   * Sentences containing complex conjunct letters (e.g., "चिकित्सकों", "सत्यापन") or Urdu-derived loanwords with nukta diacritics (क़, ख़, ग़, ज़, फ़, ड़, ढ़) cause alignment shifts. Causal cross-entropy attention maps are unable to map these complex script sequences to discrete audio frames, resulting in dropped letters, slurred syllables, or trailing silent loops.

## Decision
The zero-shot base model is completely unusable for Hindi speech generation. A dedicated, large-scale supervised fine-tuning (SFT) phase is required to align Devanagari script with Hindi pronunciation.

## Next Stage
Proceed to **Stage 1 (Pronunciation Fine-Tuning)** utilizing a filtered 31,394-sample Devanagari dataset.
