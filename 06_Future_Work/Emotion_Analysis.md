# Emotion & Speaker Profile Analysis

This document describes the methodology for analyzing emotional richness, prosodic variety, and speaker consistency in the generated speech.

---

## Objective
The objective is to establish a rigorous evaluation protocol to measure the quality of emotional expression, naturalness, and speaker consistency of the synthesized speech, moving beyond basic transcription error rates (WER/CER).

## Problem
Standard evaluation metrics like Word Error Rate (WER) and Character Error Rate (CER) are **content-focused**:
1. They measure *what* the model said, not *how* the model said it.
2. A model can achieve a perfect 0% WER while speaking in a completely flat, monotone, or robotic voice, or using the wrong speaker's voice.
3. Subjective aspects like "emotional intensity" or "natural flow" cannot be captured by standard edit-distance string comparisons.

## Hypothesis
A hybrid evaluation framework combining **acoustic feature extraction**, **neural classifier validation**, and **subjective human listening tests** will provide a complete quality score. Specifically:
* **Pitch and Energy Variance** will correlate with emotional intensity.
* **Emotion Classification Accuracy** (via an external model) will verify style distinctiveness.
* **Speaker Verification Cosine Similarity** will prove speaker identity preservation.

---

## Dataset
Evaluations are conducted on the **1,000 held-out evaluation samples** from the Rasa-emotion dataset. Each sample is tagged with a target speaker (`prerna` or `prakhar`) and one of the **7 target emotions**.

## Method

### 1. Objective Acoustic Feature Extraction
We extract physical speech features from the generated waveforms:
* **Pitch ($F_0$) Contours**: Using `librosa` or `praat-parselmouth` to map the fundamental frequency ($F_0$). Expressive emotions like *anger* or *surprise* should show higher standard deviations in pitch compared to *neutral*.
* **RMS Energy (Volume)**: Measure energy envelopes to check for natural dynamic ranges.
* **Speech Rate (Tempo)**: Compute the average phone duration. Emotional states like *sadness* should show lower tempo, while *excitement* or *anger* should show faster tempo.

### 2. Neural Classifier Validation
Synthesized audio files are passed through external, pre-trained neural networks:
* **Emotion Classifier (wav2vec2-ER)**: We run the synthesized waveform through `superb/wav2vec2-base-superb-er` and compare the predicted category against the input prompt style tag.
  $$\text{Emotion Accuracy} = \frac{\text{Count of Correct Emotion Matches}}{\text{Total Evaluation Samples}}$$
* **Speaker Verifier (ECAPA-TDNN)**: We extract speaker embeddings and check cosine similarity against the original gold speaker profile:
  $$Sim = \frac{e_{gen} \cdot e_{gold}}{\|e_{gen}\| \|e_{gold}\|}$$
  A stable speaker profile should achieve an average similarity score $> 0.75$.

### 3. Subjective Human Evaluation (MOS)
Conduct double-blind listening tests with native Hindi speakers:
* **Mean Opinion Score (MOS) for Naturalness**: Evaluators score clips from 1 (very robotic) to 5 (very natural).
* **Speaker Similarity MOS**: Evaluators listen to a reference gold clip and a generated clip, scoring voice match from 1 (different speaker) to 5 (identical speaker).
* **Emotion Intensity Scoring**: Evaluators rate if the target emotion is *weak*, *appropriate*, or *exaggerated*.

---

## Training Configuration
Not applicable (post-training analysis).

---

## Evaluation Results (Target Performance Indicators)
The target acoustic bounds for successful emotion and speaker injection are defined below:

| Target Emotion | Pitch Std Dev ($F_0$ SD) | Speech Tempo (syllables/sec) | Target Classifier Prob | Target Spk Similarity |
| :--- | :---: | :---: | :---: | :---: |
| **Neutral** | $\pm 10$ Hz | 4.5 – 5.0 | $> 0.75$ | $> 0.80$ |
| **Anger** | $\pm 35$ Hz | 5.5 – 6.2 | $> 0.70$ | $> 0.75$ |
| **Surprise** | $\pm 45$ Hz | 5.2 – 5.8 | $> 0.65$ | $> 0.75$ |
| **Sadness** | $\pm 8$ Hz | 3.5 – 4.0 | $> 0.68$ | $> 0.80$ |

---

## Limitations
* **Classifier Bias**: The `superb/wav2vec2-base-superb-er` classifier is trained on standard datasets and might struggle with accented regional Hindi dialects, introducing classification noise.
* **Human Subjectivity**: Human MOS scores exhibit high variance based on listener demographics and audio hardware quality.

## Decision
Automate the objective acoustic extraction and classifier steps into the post-training deployment pipeline, and run quarterly subjective MOS panels with 20+ native speakers to calibrate the neural rewards.

## Next Stage
Compile and summarize the **Evaluation Metrics and Results** from Baseline to Stage 3.
