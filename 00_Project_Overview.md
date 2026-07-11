# Detailed Project Presentation Slides: Orpheus TTS Hindi

This document contains a comprehensive 19-slide presentation deck mapping the Orpheus TTS Hindi project.

---

## Slide 1: Title & Executive Summary
### Project Mission: Orpheus TTS Hindi (3B)
* **Goal**: Build a high-fidelity, expressive, multi-speaker Text-to-Speech system for Hindi and Hinglish.
* **Core Backbone**: 3B parameter Llama-based Transformer, fine-tuned with parameter-efficient adapters.
* **Paradigm**: Causal, autoregressive next-token prediction over interleaved text and discrete audio tokens.
* **Development Flow**:
  * **Architecture**: Text prompt to discrete SNAC audio token mapping.
  * **Baseline**: Characterize the zero-shot baseline on Hindi.
  * **Stage 1 (Pronunciation)**: Stabilize pronunciation and solve training stability crashes.
  * **Stage 2 (Speaker ID)**: Inject consistent male (`prakhar`) and female (`prerna`) voices.
  * **Stage 3 (Emotion)**: Inject 7 distinct emotional reading styles.
  * **Future (GRPO)**: Optimize prosody and expressiveness using Reinforcement Learning.

---

## Slide 2: Architecture Core — Causal Text-to-Speech Flow
### Causal Next-Token Audio Prediction
* **Dual-Stage System Flow**:
  $$\text{Text Prompt} \xrightarrow{\text{vLLM (3B)}} \text{Audio Tokens} \xrightarrow{\text{SNAC Decoder}} \text{24kHz Waveform}$$
* **Stage 1: Causal Language Modeling**:
  * Input is formatted with a speaker name (e.g. `prerna:बोलिए: "[normalized_text]"`).
  * Llama 3B autoregressively predicts the next audio tokens.
  * Vocabulary is expanded by **28,682 custom tokens** (SNAC codebooks + special characters).
* **Stage 2: Neural Codec Decoding**:
  * Takes generated token IDs, parses them back into hierarchical SNAC codes, and decodes them into a 24kHz raw waveform.
* **Why use an LLM?**: Emergent context-awareness provides natural intonation, rhythm, and emotion without hand-crafted acoustic rules.

---

## Slide 3: Codec Deep Dive — SNAC Tokenization & Reconstructive Flow
### Multi-Scale Codec Token mapping & Streaming
* **Hierarchical Codebooks**: SNAC compresses 24kHz audio into three temporal layers:
  * **Level 1 (Coarse/Prosody)**: 12Hz (1 code/frame).
  * **Level 2 (Medium Features)**: 24Hz (2 codes/frame).
  * **Level 3 (Fine Details)**: 48Hz (4 codes/frame).
* **7-Token Frame Interleaving Pattern**:
  $$\text{Frame } i = [Q_0(i), Q_1(2i), Q_2(4i), Q_2(4i+1), Q_1(2i+1), Q_2(4i+2), Q_2(4i+3)]$$
* **Token ID to Code Mapping**:
  $$\text{SNAC Code} = \text{Token ID} - 128266 - (\text{position} \pmod 7 \times 4096)$$
* **Streaming Decoder Flow**:
  * Buffers incoming tokens until **28 tokens (4 frames)** are ready.
  * Slices samples `[2048:4096]` from the decoded waveform using overlap-add windowing to eliminate edge artifacts.
  * Yields **~85ms audio chunks** with a low latency of **~200ms to first chunk**.

---

## Slide 4: Unified Sequence Format & Generation Parameters
### Training Formatting and Inference Configuration
* **Sequence Format**: Interleaved prompt wrapped in special boundary tokens:
  `[SOH] text tokens [EOT] [EOH] [SOA] [SOS] audio tokens [EOS] [EOA]`
  * Special boundary IDs: SOH (`128259`), EOT (`128009`), EOH (`128260`), SOA (`128261`), SOS (`128257`), EOS/Stop (`128258` or default `49158`), EOA (`128262`).
* **vLLM Sampling Parameters (Inference Quality Control)**:
  * **Temperature** = `0.6` (moderate randomness; too low = robotic, too high = glitches).
  * **Top-p** = `0.8` (nucleus sampling for natural voice variations).
  * **Repetition Penalty** = `1.3` (**critical**: prevents audio token repetition and loop stutters).
  * **Max Tokens** = `1200` (caps maximum audio clip length).
* **LoRA SFT Hyperparameters**:
  * Rank $r=64$, scaling $\alpha=64$, targeting attention and MLP query matrices.
  * Optimizations: `adamw_8bit` optimizer and `use_gradient_checkpointing = "unsloth"`.

---

## Slide 5: Baseline Evaluation — Objectives & Datasets
### Zero-Shot Hindi Characterization
* **Objective**: Evaluate the zero-shot Hindi speech generation capability of the English pre-trained model.
* **Benchmark Evaluation Dataset**: 500 hand-curated samples across three phases:
  * **Phase 1: Basic Pronunciation (50 Samples)**: Easy sentences representing standard Devanagari phonology.
  * **Phase 2: Linguistic Variation (200 Samples)**: Conversational and dialect variations.
  * **Phase 3: Real World Content (250 Samples)**: Extracted sentences from news, books, and AI4Bharat datasets.
* **Text Statistics**: Average length of 20 words, 100 characters per prompt.
* **Sources**: Rasa_AI4Bharat, IndicVoices, Shekharmeena_TTS, Books, and News.

---

## Slide 6: Baseline Evaluation — Workflow & Zero-Shot Results
### Zero-Shot Baseline Performance
* **ASR Verification Loop**: Audio is synthesized from prompts -> Transcribed via `whisper-large-v3` -> WER & CER computed via `jiwer`.
* **Zero-Shot Evaluation Metrics**:
  * **Average WER (Before Norm)**: `0.9261`
  * **Average WER (After Norm)**: `0.8524`
  * **Average CER (Before Norm)**: `0.9085`
  * **Average CER (After Norm)**: `0.8351`
* **Observations & Failure Patterns**:
  * **Long Sentences (Phase 3)**: Model suffers from attention drift over long contexts, causing audio cut-offs or silent loops (average Phase 3 WER: `1.30`).
  * **Domain Mismatch**: Storytelling (WER `1.58`) and Government (WER `0.81`) show phonetic collapse on rare/formal terms.
  * **Numerals/Symbols**: Attempts to spell out raw digits as separate characters/English names rather than Hindi words.
  * **Conjuncts & Nuktas**: Complex letters (e.g. चिकित्सकों) and Urdu-derived nuktas shift alignments, dropping syllables.
* **Verdict**: Zero-shot baseline is entirely unusable. Supervised Fine-Tuning (SFT) is required.

---

## Slide 7: Stage 1 — Pronunciation Tuning Objectives
### Hindi Pronunciation Alignment SFT
* **Objective**: Establish accurate Devanagari character-to-speech token alignments, teach the model standard Hindi pronunciation, and resolve training crashes.
* **Initial Setup**: 85,587 samples from Kathbath, Indic Voices, Rasa, and Ayushi Agarwal Hinglish datasets preprocessed to 42,851 samples.
* **Initial Run Plan**: Fine-tune the 3B model for 5 epochs on Colab/Kaggle environments.
* **Result**: The initial run crashed with **NaN loss** and suffered from severe VRAM overflows.

---

## Slide 8: Stage 1 — Forensic Failure Analysis
### Understanding the "Epoch Reset" NaN Crash
* **Failure 1: Learning Rate (LR) Reset Bug**:
  * Resuming from a checkpoint while altering `num_train_epochs` (from 1 to 3) caused the trainer to discard the existing LR schedule.
  * The learning rate spiked from `0.000025` to `0.000171`, destroying the model's learned weights.
* **Failure 2: Exploding Gradients (NaN Loss)**:
  * Adjusting epochs at step 3500 without gradient clipping (`max_grad_norm`) caused gradients to grow exponentially.
  * By step 4600, the weights overflowed numerical limits, triggering the NaN crash.
* **Failure 3: VRAM OOM Outages**:
  * Unconstrained audio file durations (>15 seconds) resulted in sequence lengths exceeding the 2048 token boundary, overflowing GPU memory.

---

## Slide 9: Stage 1 — Dataset Preprocessing Filters
### Audio and Text Normalization Controls
* **Data Duration Filter**: Restrained audio segments strictly to **2.0s – 15.0s**.
* **Quality Filters**: Enforced minimum SNR $\ge$ 8.0 dB and maximum clipping fraction $\le$ 0.01.
* **Loudness Normalization**: Standardized all clips to **-23.0 LUFS** via `pyloudnorm`.
* **Devanagari Normalizer (`HindiNormalizer`)**:
  * Converted digits to words (e.g. "१२" $\rightarrow$ "बारह").
  * Composed base + nukta characters (e.g., standardizing ड़ and ढ़) and resolved conjunct anusvaras.
  * Expanded dates, currencies, and mathematical symbols.
  * Stripped stray punctuation and cleaned whitespaces.

---

## Slide 10: Stage 1 — Retraining & SFT Results
### Re-establishing Pronunciation Stability
* **Retraining Dataset**: **31,394 samples (approx. 52 hours of data)**.
  * Composition: 20% Hinglish, 10% Kathbath, 50% Male Rasa.
* **Stability Configurations (Critical Fixes)**:
  * Locked `num_train_epochs = 4` at step 0; did not alter parameters on resume.
  * Enforced gradient clipping: `max_grad_norm = 0.3`.
  * Precision: `fp16 = True`.
* **Stage 1 Results (500 Benchmark Samples)**:
  * **Average WER (After Norm)**: `0.6805` (**+20.1% improvement** vs. Base `0.8524`)
  * **Average CER (After Norm)**: `0.4790` (**+42.6% improvement** vs. Base `0.8351`)
* **Limitation**: Voice characteristics drift randomly across prompts.

---

## Slide 11: Stage 2 — Speaker Identity Objectives
### Speaker-Conditioned Voice Synthesis
* **Objective**: Inject distinct, stable, and consistent speaker profiles into the model.
* **Problem**: Stage 1 produced clear speech, but suffered from **voice drift**:
  * The voice pitch, gender, and accent drifted within sentences.
  * The user could not specify a target voice (e.g., Male vs. Female).
* **Hypothesis**: Prepending a distinct speaker name prefix to the text prompt (e.g., `SpeakerName: text`) will force self-attention heads to associate specific vocal qualities (pitch, timbre, speed) with the speaker token. Causal next-token prediction will then lock the generation to that voice profile.

---

## Slide 12: Stage 2 — Speaker SFT Methodology & Results
### Gender Mappings and Speaker Metrics
* **Dataset**: Rasa tagged dataset with explicit speaker mappings:
  * **Male voice** tagged as speaker `prakhar`.
  * **Female voice** tagged as speaker `prerna`.
* **Prompt Format**: `prakhar: बोलिए : "[text]"` or `prerna: बोलिए : "[text]"`.
* **SFT Retraining**: 4 epochs using Stage 1 checkpoint as the base.
* **Stage 2 Results (500 Benchmark Samples)**:
  * **Average WER (After Norm)**: `0.3304` (**+61.2% overall improvement** vs. Base)
  * **Average CER (After Norm)**: `0.1707` (**+79.5% overall improvement** vs. Base)
* **Takeaway**: Constraining the vocal search space to two profiles dropped the WER by **over 51% relatively** vs. Stage 1, proving speaker stability directly enhances pronunciation clarity.
* **Limitation**: Speech is monotone and lacks emotional expression.

---

## Slide 13: Stage 3 — Emotion Injection Objectives
### Prosodic and Style Control
* **Objective**: Enable joint control of speaker identity and emotional reading styles.
* **Problem**: SFT models from Stage 2 are clear and consistent but speak in a flat, monotone voice, lacking natural intonation, tempo variation, and prosody.
* **Hypothesis**: Fine-tuning on a dataset tagged with both speaker and emotional style will teach the model joint representations, allowing concurrent conditioning at inference:
  $$\text{Prompt} = \text{"User: <[speaker]> <[style]> बोलिए : \"[text]\""}$$

---

## Slide 14: Stage 3 — Joint SFT & Metrics Results
### 7-Emotion SFT Performance
* **Dataset**: Balanced emotional corpus of **5,000 samples** representing **7 emotions** (anger, surprise, sadness, etc.) mapped across `prerna` and `prakhar`.
* **Sampling**: Applied stratified sampling with replacement to minority styles to prevent mode collapse.
* **Stage 3 Results (500 Benchmark Samples)**:
  * **Average WER (After Norm)**: `0.3093` (**+63.7% overall improvement** vs. Base)
  * **Average CER (After Norm)**: `0.1544` (**+81.5% overall improvement** vs. Base)
* **Takeaway**: Stage 3 pushed WER down an additional **~2.1%** and CER down **~1.6%** compared to Stage 2, confirming that style tags act as regularizers to improve alignments.
* **Limitation**: Models are bound to static labels; cannot adjust emotional intensity.

---

## Slide 15: Future Work — GRPO RL Objectives
### Reinforcement Learning for Emotional Enrichment
* **Objective**: Apply Group Relative Policy Optimization (GRPO) to fine-tune the Stage 3 SFT model for natural prosody, vocal clarity, and emotional expression.
* **Problem**: Standard RLHF (PPO) requires a memory-heavy Critic network, which overflows VRAM on dual-GPU setups.
* **Mathematical Solution**: GRPO estimates the baseline directly from a group of $G=4$ parallel rollout completions, eliminating the Critic network and saving 50% VRAM:
  $$A_i = \frac{R(q, o_i) - \text{mean}(\{R_1, \dots, R_G\})}{\text{std}(\{R_1, \dots, R_G\}) + \epsilon}$$
* **Dataset**: Rasa dataset of 5,000 samples (80% Active Emotional, 20% Neutral Anchor to prevent style collapse).

---

## Slide 16: Future Work — Multi-Objective Reward Engine
### Scoring Auditory Outputs
* **Objective**: Score generated audio rollouts using three automated engines running on `cuda:1` to guide the policy:
1. **Phonetic Accuracy ($R_{ASR}$)**: Whisper-large-v3 transcribes the audio, `HindiNormalizer` cleans the text, and WER/CER are computed:
   $$R_{WER} = 1.0 - \tanh(3.0 \times WER) \quad R_{CER} = 1.0 - \tanh(3.0 \times CER)$$
   $$R_{ASR} = 0.7 \times R_{WER} + 0.3 \times R_{CER}$$
2. **Speaker Consistency ($R_{spk}$)**: ECAPA-TDNN extracts embeddings; cosine similarity is calculated against the gold profile (`prerna.wav`/`prakhar.wav`):
   $$R_{spk} = \text{clamp}\left(\frac{Sim - 0.2}{0.6}, 0.0, 1.0\right)$$
3. **Emotion Match ($R_{emo}$)**: wav2vec2-ER classifier computes the target category probability.

---

## Slide 17: Future Work — Split-GPU Optimization
### Balancing Policy and Reward VRAM
* **Problem**: Standard execution of Policy training and sequential evaluation on the same GPU crashes due to VRAM OOM.
* **Solution**: Implement a Split-GPU architecture across 2 × 15 GB GPUs:
  * **CUDA:0 (Policy & Gradient Updates)**:
    * Policy model loaded in fp16.
    * LoRA adapters ($r=64$, $\alpha=64$).
    * Total VRAM: **~7.5 GB** (with gradient checkpointing).
  * **CUDA:1 (Automated Rewards)**:
    * Whisper-large-v3: ~3.1 GB.
    * ECAPA-TDNN: ~200 MB.
    * wav2vec2-ER: ~400 MB.
    * Total VRAM: **~3.7 GB** (running sequentially).

---

## Slide 18: Future Work — Backbone Replacements
### Downscaling the Backbone: Gemma-3-270M and Qwen3-4B
* **Objective**: Evaluate alternative transformer backbones to optimize latency and memory footprint.
* **Backbone Upgrades**:
  * **Gemma-3-270M (Edge Deployment)**: Replace the 3B Llama backbone with a 270M parameter model to decrease VRAM requirements to <1 GB, enabling edge/CPU execution.
  * **Qwen3-4B / Gemma-3-4B (Vocab Optimization)**: Larger multilingual vocabularies compress Hindi text sequences by 30-40%, allowing longer audio rollouts.
* **Method**: Load backbone -> register special boundary tokens -> target LoRA modules -> perform Stages 1-3 SFT -> evaluate WER/CER and throughput.

---

## Slide 19: Future Work — Emotion & Quality Analysis
### Quantifying Subjective Characteristics
* **Objective**: Build evaluation metrics for prosodic naturalness, pitch variance, and voice profiles post-training.
* **Evaluation Framework**:
  1. **Acoustic Contours**: Extract pitch ($F_0$) standard deviations, RMS energy envelopes, and syllable durations using `praat-parselmouth` and `librosa`.
  2. **Classifier Accuracy**: Track average emotion probability from wav2vec2-ER and speaker verification cosine similarity on 1,000 held-out samples.
  3. **Mean Opinion Score (MOS)**: Run quarterly blind listening tests with native speakers scoring naturalness and speaker match on a 1-5 scale.
