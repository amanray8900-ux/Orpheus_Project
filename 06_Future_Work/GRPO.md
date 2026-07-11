# RLHF & Group Relative Policy Optimization (GRPO)

This document describes the Reinforcement Learning (RL) training design for Orpheus TTS Hindi using Group Relative Policy Optimization (GRPO).

---

## Objective
The objective is to apply Reinforcement Learning to optimize the SFT model (Stage 3) for vocal clarity, speaker profile compliance, and rich emotional expressiveness without exceeding VRAM limitations.

## Problem
Standard RLHF utilizing PPO (Proximal Policy Optimization) requires training a separate Critic (Value) network. Because the Critic network is typically of the same parameter scale as the Policy network (~3B parameters), running PPO:
1. Doubles the GPU memory footprint.
2. Severely limits the sequence rollout batch size.
3. Cannot fit on consumer-grade dual-GPU setups (like dual-T4 GPUs with 15 GB VRAM each).

## Hypothesis
1. **Group Advantage Estimation**: By replacing the Critic network with the average reward of a group of $G=4$ parallel rollout completions per prompt, Critic memory overhead is reduced to zero, saving up to 50% of VRAM.
2. **Multi-Objective Rewards**: Guide the policy through a composite reward engine (Whisper ASR, ECAPA-TDNN Speaker Encoder, and wav2vec2-ER Emotion Classifier) running sequentially on a secondary GPU to avoid memory overflow.
3. **Catastrophic Style Forgetting Guard**: Enforcing an 80/20 active-emotional to neutral-anchor sampling ratio will prevent prosodic collapse (flattening of the voice) during RL.

---

## Dataset & Sampling Strategy
* **Source**: Rasa Hindi dataset.
* **Size**: 5,000 samples (1,000 samples held out for evaluation).
* **Speaker Mapping**:
  * Gender: Male $\rightarrow$ mapped to `prakhar`
  * Gender: Female $\rightarrow$ mapped to `prerna`
* **Prompt Format**:
  `User: <[speaker]> <[style]> बोलिए : "[normalized_text]"`
  `Assistant: <|SPEECH_GENERATION_START|>`
* **80 / 20 Style Ratio Strategy**:
  * **20% (1,000 Samples) Neutral Anchor Group**: Includes BOOK, INDIC CONV, WIKI, NEWS, ALEXA, UNKNOWN, and NORMAL styles. This acts as an anchor to prevent the model from forgetting neutral pronunciations.
  * **80% (4,000 Samples) Active Emotional Group**: Expressive styles (ANGER, SURPRISE, etc.) sampled using stratified sampling with replacement to ensure equal emotional weight.

---

## Method: GRPO Mathematics

### 1. Group Advantage Calculation
For each prompt $q$ sampled from the dataset, the Policy model ($\pi_\theta$) generates a group of $G=4$ independent audio completions $\{o_1, o_2, o_3, o_4\}$. The reward engine evaluates each completion, returning a reward score $R(q, o_i)$.

The advantage ($A_i$) of completion $o_i$ relative to the group is computed as:
$$A_i = \frac{R(q, o_i) - \text{mean}(\{R(q, o_1), \dots, R(q, o_G)\})}{\text{std}(\{R(q, o_1), \dots, R(q, o_G)\}) + \epsilon}$$
*Intuition*: Completions that sound better than the group average get positive advantages and are reinforced; completions that sound worse get negative advantages and are discouraged.

### 2. GRPO Objective Function
The objective is to maximize the clipped policy gradient loss with a KL-divergence penalty to prevent the model from drifting too far from the Stage 3 SFT reference model ($\pi_{ref}$):
$$\mathcal{L}_{GRPO}(\theta) = \frac{1}{G} \sum_{i=1}^G \left[ \min\left(r_i(\theta) A_i, \text{clip}(r_i(\theta), 1-\epsilon, 1+\epsilon) A_i\right) - \beta D_{KL}(\pi_\theta \parallel \pi_{ref}) \right]$$
Where:
* $r_i(\theta) = \frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{old}}(o_i \mid q)}$ is the probability ratio.
* $\epsilon = 0.2$ is the clipping parameter.
* $\beta = 0.01$ is the KL penalty coefficient.
* $D_{KL}(\pi_\theta \parallel \pi_{ref}) = \frac{\pi_{ref}(o_i \mid q)}{\pi_\theta(o_i \mid q)} \log \left(\frac{\pi_{ref}(o_i \mid q)}{\pi_\theta(o_i \mid q)}\right) - 1$ is approximated online per-token.

---

## Multi-Objective Reward Architecture
The reward engine runs on the secondary GPU (`cuda:1`) and sequentially executes three scoring steps:

```
Generated SNAC Tokens (cuda:0)
        │
        ▼ (snac_intercept.py)
   24kHz Waveform
        │
        ▼ (sent to cuda:1)
┌────────────────────────────────────────────────────────────┐
│                  EVALUATION ENGINE (cuda:1)                │
├─────────────────────┬───────────────────┬──────────────────┤
│    Whisper ASR      │    ECAPA-TDNN     │    wav2vec2-ER   │
│   (WER/CER loss)    │ (Speaker cosine)  │ (Emotion probability)
└──────────┬──────────┴─────────┬─────────┴────────┬─────────┘
           │                    │                  │
           ▼                    ▼                  ▼
      Reward R_ASR         Reward R_spk       Reward R_emo
           │                    │                  │
           └────────────────────┼──────────────────┘
                                │
                                ▼
                       Combined RL Reward (R)
```

### A. Phonetic Accuracy Reward ($R_{ASR}$)
Transcribes the waveform using `openai/whisper-large-v3` set to Hindi, runs the Devanagari normalizer (`HindiNormalizer`), and computes Word Error Rate (WER) and Character Error Rate (CER).
$$R_{WER} = 1.0 - \tanh(3.0 \times WER)$$
$$R_{CER} = 1.0 - \tanh(3.0 \times CER)$$
$$R_{ASR} = 0.7 \times R_{WER} + 0.3 \times R_{CER}$$

### B. Speaker Identity Reward ($R_{spk}$)
Resamples the waveform to 16kHz and extracts a speaker embedding using `speechbrain/spkrec-ecapa-voxceleb`. Cosine similarity is computed against the gold reference embedding ($e_{gold}$) of the target speaker profile (`prakhar.wav` or `prerna.wav`):
$$Sim = \frac{e_{gen} \cdot e_{gold}}{\|e_{gen}\| \|e_{gold}\|}$$
The reward is normalized and clamped over a typical target variance range:
$$R_{spk} = \text{clamp}\left(\frac{Sim - 0.2}{0.6}, 0.0, 1.0\right)$$

### C. Emotion / Style Reward ($R_{emo}$)
Evaluates emotional expressiveness using the `superb/wav2vec2-base-superb-er` classifier. Manifest styles are mapped to classifier labels:
* ANGER $\rightarrow$ *angry*
* SURPRISE $\rightarrow$ *surprise*
* BOOK / CONV / WIKI / NEWS / BB / INDIC $\rightarrow$ *neutral*
The reward is the classifier's probability of the target mapped category:
$$R_{emo} = \text{Prob}(\text{Target Category} \mid \text{Generated Waveform})$$

---

## Training & Hardware Configuration
To run this setup on dual-T4 GPUs (2 × 15 GB VRAM) without memory failures, a **Split-GPU memory model** is deployed:

* **CUDA:0 — Policy & Optimizer (~7.5 GB VRAM)**:
  * Policy model: Orpheus 3B loaded in fp16.
  * Trainable parameter LoRA adapters: ~6.5 GB.
  * Handles forward rollouts and backward gradient optimization steps.
* **CUDA:1 — Sequential Reward Evaluation (~3.7 GB VRAM)**:
  * Whisper-large-v3: ~3.1 GB.
  * ECAPA-TDNN: ~200 MB.
  * wav2vec2-ER: ~400 MB.
  * Running sequentially prevents VRAM spikes on the secondary card.

## Evaluation
Verified on the 1,000 held-out samples by measuring:
1. Stability of ASR transcription scores ($R_{ASR}$).
2. Cosine similarity of the voice signature ($R_{spk}$).
3. Dynamic range and intensity of the target emotion probability ($R_{emo}$).

---

## Limitations
1. **ASR Robustness**: Whisper-large-v3 transcriptions are subject to ASR hallucinations, which can occasionally output incorrect zero-reward scores.
2. **Inference Latency during Rollouts**: Autoregressive sampling of $G=4$ candidates per prompt is slow, representing the primary training bottleneck.

## Decision
Implement the Split-GPU GRPO model with the 80/20 active-neutral sampling strategy for the post-SFT reinforcement learning phase.

## Next Stage
Investigate model size reduction and efficiency via **LLM Backbone Replacement** research.
