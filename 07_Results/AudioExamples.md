# Audio Verification and Examples

This document outlines the pipeline and criteria for curating, compiling, and checking representative audio examples from each training checkpoint.

---

## Objective
The objective is to establish a standardized procedure for generating, saving, and auditing audio examples to qualitatively verify the clarity, speaker consistency, and emotional expression of the synthesized speech.

## Problem
Relying solely on objective text-based scores (WER/CER) is insufficient for audio assessment:
1. **ASR Insensitivity**: Transcription models like Whisper can successfully transcribe words even if the underlying audio has severe background hiss, clicking, clipping, or digital glitches.
2. **Acoustic Nuance**: Natural flow, pause placements, breath sounds, and dynamic range are critical to human-like speech but are completely ignored by character edit-distance scores.

## Hypothesis
Curation of a static, representative "Auditory Benchmark Suite" of 20 text prompts—representing varying phonetic difficulties, speaker tag mappings, and emotional styles—will provide a quick, qualitative audit loop to detect model regressions.

---

## Dataset: Auditory Benchmark Suite
The suite comprises 20 selected text prompts representing:
* **Pronunciation Test Set**: 5 Devanagari prompts containing complex conjuncts, nasal sounds (anusvara), and nuktas (e.g., "जेसीओ", "चिकित्सकों", "सत्यापन", "डॉट", "ड़/ढ़").
* **Speaker Match Set**: 5 prompts tested on both `prerna` and `prakhar` tags.
* **Emotion Match Set**: 10 prompts tested across the 7 emotional style tags (e.g., `<anger>`, `<surprise>`, `<sadness>`).

---

## Method: Generation & Verification Pipeline
1. **Native Optimization**: Set the model to inference mode using Unsloth's native 2x faster inference:
   ```python
   FastLanguageModel.for_inference(model)
   ```
2. **Sequence Padding**: Format prompts into sequence frames and pad them correctly:
   ```python
   # Start of human (128259) + encoded text + End of text (128009) + End of human (128260)
   modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)
   ```
3. **Autoregressive Generation**: Run `model.generate` with standard hyperparameters:
   ```python
   generated_ids = model.generate(
       input_ids=input_ids,
       attention_mask=attention_mask,
       max_new_tokens=1200,
       do_sample=True,
       temperature=0.6,
       top_p=0.95,
       repetition_penalty=1.1,
       eos_token_id=128258
   )
   ```
4. **SNAC Decoding**: Extract generated codes, subtract vocabulary offsets, reconstruct the 3 hierarchical arrays, and run `snac_model.decode()` to output a 24kHz waveform.
5. **Auditory Audit Checklist**:
   * **Phonetic Clarity**: Are syllables clear? Is there any slurring or clipping?
   * **Gender Consistency**: Does `prerna` remain consistently female, and `prakhar` consistently male?
   * **Prosody Control**: Does the `<surprise>` tag generate higher pitch fluctuations?
   * **Glitches**: Is the audio free of digital hums, clipping, or looping?

---

## Training Configuration
Not applicable (post-training generation setup).

---

## Results
The qualitative performance changes observed across stages are detailed below:

* **Base Model**: Synthesizes highly noisy, slurred sounds. Devanagari text prompts fail to align.
* **Stage 1 (Pronunciation)**: Sounds like intelligible Hindi speech, but the voice sounds like a generic robot and occasionally shifts identity mid-sentence.
* **Stage 2 (Speaker ID)**: Synthesizes consistent and clear voices for `prerna` and `prakhar`. Pronunciation is clean, but the tone is flat and monotone.
* **Stage 3 (Emotion)**: Introduces distinct prosodic differences. An `<anger>` tag results in a faster, high-energy voice; `<surprise>` shows pitch surges; `<neutral>` remains flat and clear.

---

## Limitations
* **Subjective Variance**: Listening tests are subjective and can vary based on the listener's environment and audio equipment.
* **Autoregressive Hallucinations**: Due to sampling temperature, the same prompt can occasionally generate high-quality audio in one run and a glitched audio in the next (requiring multiple seed checks).

## Decision
Save generated waveforms of the 20 benchmark prompts under a permanent directory after each major SFT or RL checkpoint is compiled, ensuring a qualitative record of model progression.

## Next Stage
This completes the SFT documentation sequence. The next stage of model optimization will focus on reinforcement learning (GRPO).
