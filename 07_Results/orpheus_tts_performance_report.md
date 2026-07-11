# Orpheus Hindi TTS Model — Performance Evaluation Report

> **Evaluation Framework**: Whisper-large ASR | **Language**: Hindi (hi-IN) | **Total Samples**: 500  
> **Report Date**: July 2026

---

## 1. Executive Summary

This report presents a comprehensive evaluation of the **Orpheus Hindi TTS model** across four training stages:

| Stage | Description |
|---|---|
| **Base Model** | Pre-trained Orpheus TTS without Hindi-specific fine-tuning |
| **Stage 1** | Fine-tuned on Hindi speech data for pronunciation accuracy |
| **Stage 2** | Further optimized for speaker characteristics and naturalness |
| **Stage 3** | Additional fine-tuning pass evaluated across all three evaluation phases |

All 500 evaluation samples were synthesized by each model stage, then transcribed by **Whisper-large ASR**. The transcriptions were compared against normalized reference texts using **Word Error Rate (WER)** and **Character Error Rate (CER)**.

### Key Findings at a Glance

| Metric | Base Model | Stage 1 | Stage 2 | Stage 3 | Δ Base→S3 |
|---|---|---|---|---|---|
| **Mean WER** | 0.9246 | 0.3514 | 0.3304 | **0.3093** | ✅ **−66.6%** |
| **Mean CER** | 0.8801 | 0.1913 | 0.1707 | **0.1549** | ✅ **−82.4%** |
| **Median WER** | 0.3333 | 0.3060 | 0.2857 | **0.2632** | ✅ **−21.0%** |
| **Median CER** | 0.2330 | 0.1176 | 0.1111 | **0.1040** | ✅ **−55.4%** |

> [!IMPORTANT]
> **Fine-tuning reduced mean WER by 66.6% and mean CER by 82.4%** from the Base Model to Stage 3. Stage 3 delivers a further 6.4% WER and 9.3% CER improvement over Stage 2. The large gap between mean and median in the Base Model (WER: 0.92 mean vs 0.33 median) indicates a long tail of high-error samples — particularly on short texts where ASR hallucinations inflate scores. All three fine-tuning stages progressively compressed this tail.

### Sample-Level Improvement Distribution (Base → Stage 2, WER)

| Outcome | Count | Percentage |
|---|---|---|
| ✅ Improved | 291 | 58.2% |
| ⚠️ Same | 14 | 2.8% |
| ❌ Regressed | 195 | 39.0% |

---

## 2. Stage-over-Stage Improvement Analysis

This section shows the **absolute and relative improvement** at each transition.

### Base Model → Stage 1

| Metric | Base | Stage 1 | Absolute Δ | Relative Δ |
|---|---|---|---|---|
| Mean WER | 0.9246 | 0.3514 | −0.5732 | **−62.0%** |
| Mean CER | 0.8801 | 0.1913 | −0.6888 | **−78.3%** |

- 275 samples improved (55.0%), 211 regressed (42.2%), 14 same (2.8%)

### Stage 1 → Stage 2

| Metric | Stage 1 | Stage 2 | Absolute Δ | Relative Δ |
|---|---|---|---|---|
| Mean WER | 0.3514 | 0.3304 | −0.0210 | **−6.0%** |
| Mean CER | 0.1913 | 0.1707 | −0.0206 | **−10.8%** |

### Stage 2 → Stage 3

| Metric | Stage 2 | Stage 3 | Absolute Δ | Relative Δ |
|---|---|---|---|---|
| Mean WER | 0.3304 | **0.3093** | −0.0211 | **−6.4%** |
| Mean CER | 0.1707 | **0.1549** | −0.0158 | **−9.3%** |

> [!NOTE]
> Stage 1 provided the largest single improvement. Stage 2 (speaker optimization) added ~6% WER / ~11% CER gain. Stage 3 delivers a comparable additional step: −6.4% WER and −9.3% CER over Stage 2, confirming continued, consistent progress across all three fine-tuning phases.

---

## 3. Performance by Evaluation Phase

| Phase | N | WER Base | WER S1 | WER S2 | WER S3 | Δ WER (B→S3) | CER Base | CER S1 | CER S2 | CER S3 | Δ CER (B→S3) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Phase 1** — Basic Pronunciation | 50 | 0.9295 | 0.3013 | 0.2963 | **0.2836** | ✅ −69.5% | 0.6532 | 0.1464 | 0.1446 | **0.1329** | ✅ −79.7% |
| **Phase 2** — Linguistic Variation | 200 | 0.4107 | 0.3212 | 0.2705 | **0.2667** | ✅ −35.1% | 0.3059 | 0.1970 | 0.1517 | **0.1501** | ✅ −50.9% |
| **Phase 3** — Real-World Content | 250 | 1.3347 | 0.3856 | 0.3851 | **0.3486** | ✅ −73.9% | 1.3849 | 0.1957 | 0.1912 | **0.1631** | ✅ −88.2% |

![WER by Evaluation Phase × Training Stage](C:/Users/amanr/.gemini/antigravity-ide/brain/a9cb53ed-4cdd-4187-8503-06fdae93f938/plots/plot_2_wer_by_phase.png)

![CER by Evaluation Phase × Training Stage](C:/Users/amanr/.gemini/antigravity-ide/brain/a9cb53ed-4cdd-4187-8503-06fdae93f938/plots/plot_5_cer_by_phase.png)

**Observations:**
- Phase 3 (real-world content) had the highest Base Model error (WER 1.33, CER 1.38), but showed the **most dramatic improvement** after fine-tuning through Stage 3 (−73.9% WER, −88.2% CER). Stage 3 alone reduced WER by a further −9.5% and CER by −14.7% over Stage 2 in this phase.
- Phase 2 (linguistic variation) maintained the lowest Stage 3 WER (0.2667), confirming the model handles varied linguistic constructions well across all stages.
- Phase 1 (basic pronunciation) achieved the lowest Stage 3 WER (0.2836) and CER (0.1329) among all phases, confirming solid foundational pronunciation.
- **Stage 3 improved all three phases** over Stage 2 with consistent gains, indicating the additional fine-tuning pass is broadly beneficial rather than phase-specific.

---

## 4. Performance by Data Source (`source_from`)

| Source | N | WER Base | WER S1 | WER S2 | Δ WER (B→S2) | CER Base | CER S1 | CER S2 | Δ CER (B→S2) |
|---|---|---|---|---|---|---|---|---|---|
| **Hindi_Books_Audio** | 50 | 1.0167 | 0.3872 | **0.3671** | ✅ −63.9% | 0.9357 | 0.1740 | **0.1741** | ✅ −81.4% |
| **Hindi_Story_News** | 50 | 1.7274 | 0.4082 | **0.4727** | ✅ −72.6% | 1.5880 | 0.2039 | **0.2448** | ✅ −84.6% |
| **IndicVoices** | 100 | 1.7496 | 0.3513 | **0.3363** | ✅ −80.8% | 1.9851 | 0.2122 | **0.2057** | ✅ −89.6% |
| **LLM_Generated** | 80 | 0.3392 | 0.2505 | **0.2162** | ✅ −36.2% | 0.2626 | 0.1631 | **0.1288** | ✅ −51.0% |
| **Rasa_AI4Bharat** | 110 | 0.4807 | 0.2866 | **0.2876** | ✅ −40.2% | 0.3952 | 0.1173 | **0.1106** | ✅ −72.0% |
| **Shekharmeena_TTS** | 110 | 0.6373 | 0.4478 | **0.3694** | ✅ −42.0% | 0.4626 | 0.2690 | **0.1945** | ✅ −57.9% |

![WER by Data Source × Training Stage](C:/Users/amanr/.gemini/antigravity-ide/brain/a9cb53ed-4cdd-4187-8503-06fdae93f938/plots/plot_7_wer_by_source.png)

**Observations:**
- **IndicVoices** showed the highest Base Model errors but also the **greatest improvement** (−80.8% WER, −89.6% CER), suggesting the model learned to generalize from this diverse source well.
- **LLM_Generated** samples had the lowest errors across all stages, which is expected since LLM-generated text tends to be cleaner and more structurally consistent.
- **Hindi_Story_News** has a notable pattern: Stage 2 WER (0.4727) is higher than Stage 1 (0.4082), indicating **partial regression** in Stage 2 for news/story content — a potential area for targeted improvement.
- **Shekharmeena_TTS** consistently has the highest Stage 1 and Stage 2 WER among all sources, suggesting this source's voice characteristics present an ongoing challenge.

---

## 5. Performance by Script Complexity

| Complexity | N | WER Base | WER S1 | WER S2 | Δ WER (B→S2) | CER Base | CER S1 | CER S2 | Δ CER (B→S2) |
|---|---|---|---|---|---|---|---|---|---|
| **Simple** | 284 | 1.3413 | 0.3433 | **0.3213** | ✅ −76.0% | 1.3438 | 0.1808 | **0.1662** | ✅ −87.6% |
| **Moderate** | 185 | 0.3410 | 0.3378 | **0.3151** | ✅ −7.6% | 0.2413 | 0.1883 | **0.1573** | ✅ −34.8% |
| **Complex** | 28 | 0.5552 | 0.4603 | **0.4590** | ✅ −17.3% | 0.3969 | 0.2439 | **0.2288** | ✅ −42.4% |
| **Very_Complex** | 3 | 0.9087 | 0.9480 | **0.9306** | ⚠️ −2.4% | 0.8863 | 0.8863 | **0.8876** | ❌ +0.1% |

![WER by Script Complexity × Training Stage](C:/Users/amanr/.gemini/antigravity-ide/brain/a9cb53ed-4cdd-4187-8503-06fdae93f938/plots/plot_3_wer_by_complexity.png)

**Observations:**
- **Simple scripts** had the highest Base Model error (inflated by short-text hallucinations) but showed the **most dramatic improvement** (−76% WER).
- **Very Complex scripts** (N=3) remain a critical struggle zone. Fine-tuning barely improved them (WER ~0.93 across all stages), and CER actually regressed slightly. These samples involve extremely long, literary Hindi with dense conjuncts and rare vocabulary.
- **Complex scripts** (N=28) see moderate improvement (−17% WER), but Stage 1 → Stage 2 improvement is negligible, suggesting speaker optimization doesn't help with complex phonetic challenges.

> [!WARNING]
> **Struggle Zone**: Very_Complex and Complex scripts show minimal improvement from fine-tuning. The model needs targeted training on complex Hindi orthographic patterns (conjuncts, rare characters, literary vocabulary).

---

## 6. Performance by Sentence Length Category

| Category | N | WER Base | WER S1 | WER S2 | Δ WER (B→S2) | CER Base | CER S1 | CER S2 | Δ CER (B→S2) |
|---|---|---|---|---|---|---|---|---|---|
| **Short** (1–5 words) | 70 | 3.9528 | 0.4117 | **0.4617** | ✅ −88.3% | 4.3220 | 0.1606 | **0.2234** | ✅ −94.8% |
| **Medium** (6–10 words) | 183 | 0.5470 | 0.3375 | **0.3008** | ✅ −45.0% | 0.4202 | 0.1859 | **0.1530** | ✅ −63.6% |
| **Long** (11–20 words) | 129 | 0.3610 | 0.3026 | **0.2887** | ✅ −20.0% | 0.2501 | 0.1510 | **0.1268** | ✅ −49.3% |
| **Very_Long** (21+ words) | 118 | 0.3299 | 0.3908 | **0.3440** | ❌ +4.3% | 0.2404 | 0.2619 | **0.2151** | ✅ −10.5% |

**Observations:**
- **Short texts** had extremely high Base Model WER (3.95) due to ASR hallucinations — the model would generate extra words for very short inputs. Fine-tuning reduced this dramatically (−88%), but Stage 2 WER (0.46) is still the highest among all categories.
- **Medium and Long texts** show the best Stage 2 performance (WER 0.30 and 0.29 respectively), representing the "sweet spot" for the model.
- **Very_Long texts** show a **regression pattern**: Stage 1 WER (0.39) and Stage 2 WER (0.34) are both *higher* than Base Model WER (0.33). The fine-tuning seems to have introduced some degradation for very long sequences.

> [!CAUTION]
> **Regression on Very Long Texts**: WER increased by +4.3% for Very_Long sentences (Base→S2). This suggests the fine-tuning data may not have contained sufficient long-form content, causing the model to lose some of its original long-text generation capability.

---

## 7. Performance by Code Mixing Ratio

| Bucket | N | WER Base | WER S1 | WER S2 | Δ WER (B→S2) | CER Base | CER S1 | CER S2 | Δ CER (B→S2) |
|---|---|---|---|---|---|---|---|---|---|
| **0.00 (None)** | 394 | 1.0040 | 0.3346 | **0.3252** | ✅ −67.6% | 0.9885 | 0.1659 | **0.1552** | ✅ −84.3% |
| **0.01–0.10 (Low)** | 30 | 0.4127 | 0.4379 | **0.2961** | ✅ −28.2% | 0.3417 | 0.3202 | **0.1965** | ✅ −42.5% |
| **0.11–0.30 (Moderate)** | 51 | 0.4731 | 0.4079 | **0.3353** | ✅ −29.1% | 0.3922 | 0.2824 | **0.2155** | ✅ −45.0% |
| **0.31–0.50 (High)** | 22 | 1.1834 | 0.4495 | **0.3707** | ✅ −68.7% | 0.7973 | 0.2847 | **0.2265** | ✅ −71.6% |
| **0.50+ (Very High)** | 3 | 1.3935 | 0.0238 | **0.9788** | ✅ −29.8% | 0.9306 | 0.0105 | **0.7878** | ✅ −15.3% |

**Observations:**
- **No code mixing** (pure Hindi) accounts for 78.8% of samples and shows strong improvement (−67.6% WER).
- **High code mixing** (0.31–0.50) shows a dramatic improvement (−68.7% WER), comparable to pure Hindi, suggesting the model handles Hindi-English mixing well after fine-tuning.
- **Very High code mixing** (0.50+, N=3) is interesting: Stage 1 achieved near-perfect WER (0.024) but Stage 2 regressed significantly (0.979). This may be due to the small sample size or speaker optimization interfering with English-dominant synthesis.
- **Low and Moderate mixing** show smaller but consistent improvements, with Stage 2 performing better than Stage 1 in all cases.

---

## 8. Performance by Word Count (Interval Buckets)

| Word Range | N | WER Base | WER S1 | WER S2 | Δ WER (B→S2) | CER Base | CER S1 | CER S2 | Δ CER (B→S2) |
|---|---|---|---|---|---|---|---|---|---|
| **1–5** | 70 | 3.9528 | 0.4117 | **0.4617** | ✅ −88.3% | 4.3220 | 0.1606 | **0.2234** | ✅ −94.8% |
| **6–10** | 83 | 0.6520 | 0.3673 | **0.3184** | ✅ −51.2% | 0.5118 | 0.1955 | **0.1465** | ✅ −71.4% |
| **11–20** | 189 | 0.4123 | 0.3173 | **0.2912** | ✅ −29.4% | 0.2984 | 0.1693 | **0.1457** | ✅ −51.2% |
| **21–50** | 139 | 0.3126 | 0.3166 | **0.2826** | ✅ −9.6% | 0.2144 | 0.1832 | **0.1402** | ✅ −34.6% |
| **51–100** | 15 | 0.4437 | 0.5756 | **0.5564** | ❌ +25.4% | 0.3648 | 0.4790 | **0.4683** | ❌ +28.4% |
| **100+** | 4 | 0.8604 | 0.9498 | **0.9432** | ❌ +9.6% | 0.8406 | 0.8851 | **0.8811** | ❌ +4.8% |

![WER by Word Count Range × Training Stage](C:/Users/amanr/.gemini/antigravity-ide/brain/a9cb53ed-4cdd-4187-8503-06fdae93f938/plots/plot_4_wer_by_wordcount.png)

**Observations:**
- The model performs best on texts with **6–50 words** (WER 0.28–0.32 at Stage 2).
- **Very short texts (1–5 words)** show the most improvement from Base (−88%) but still have the highest Stage 2 WER among the primary ranges due to hallucination artifacts.
- **51–100 word texts** are a clear **regression zone**: Stage 2 WER (0.5564) is 25.4% *worse* than Base Model (0.4437). This is the most concerning finding.
- **100+ word texts** also regress, though with only 4 samples this finding is less statistically significant.

> [!WARNING]
> **Critical Regression**: Texts with 51–100 words show a 25.4% WER increase after fine-tuning. This represents a significant quality degradation for medium-long content and should be prioritized for investigation.

---

## 9. Performance by Character Count (Interval Buckets)

| Char Range | N | WER Base | WER S1 | WER S2 | Δ WER (B→S2) | CER Base | CER S1 | CER S2 | Δ CER (B→S2) |
|---|---|---|---|---|---|---|---|---|---|
| **1–20** | 48 | 4.7386 | 0.4031 | **0.4708** | ✅ −90.1% | 5.4732 | 0.1520 | **0.2646** | ✅ −95.2% |
| **21–50** | 109 | 0.9692 | 0.3651 | **0.3369** | ✅ −65.2% | 0.7751 | 0.1820 | **0.1434** | ✅ −81.5% |
| **51–100** | 184 | 0.4143 | 0.3150 | **0.2869** | ✅ −30.8% | 0.2999 | 0.1699 | **0.1422** | ✅ −52.6% |
| **101–200** | 125 | 0.3060 | 0.3046 | **0.2771** | ✅ −9.4% | 0.2059 | 0.1668 | **0.1296** | ✅ −37.1% |
| **200–500** | 30 | 0.3761 | 0.5583 | **0.4889** | ❌ +30.0% | 0.2857 | 0.4288 | **0.3722** | ❌ +30.2% |
| **500+** | 4 | 0.8604 | 0.9498 | **0.9432** | ❌ +9.6% | 0.8406 | 0.8851 | **0.8811** | ❌ +4.8% |

**Observations:**
- The model's **optimal range** is 51–200 characters (WER 0.28–0.29 at Stage 2), covering the majority of samples (309 out of 500).
- **Very short texts (1–20 chars)** show massive Base Model inflation (WER 4.74) but are well-corrected by fine-tuning (−90%), though Stage 2 slightly regresses from Stage 1.
- **200–500 character texts** show a concerning **30% WER regression** from Base to Stage 2. This is consistent with the word count findings and confirms a long-text degradation problem.

---

## 10. Key Findings & Discussion

### Where the Model Excels ✅
1. **Medium-length Hindi text** (6–50 words / 51–200 chars): Consistent WER < 0.30, CER < 0.15 at Stage 2.
2. **Pure Hindi without code mixing**: Strong performance with 67.6% WER improvement.
3. **LLM-Generated and Rasa_AI4Bharat content**: Lowest error rates, suggesting clean training text translates to better synthesis.
4. **Simple and Moderate script complexity**: Good handling of standard Devanagari with consistent improvement.

### Where the Model Struggles ❌
1. **Very Complex scripts** (N=3): WER ~0.93 across all stages — near-complete failure. Literary Hindi with dense conjuncts and rare vocabulary remains unsolvable.
2. **Long texts (51+ words / 200+ chars)**: Regression of 25–30% in WER after fine-tuning. The model loses coherence on extended passages.
3. **Very short texts (1–5 words)**: While improved from Base, Stage 2 WER (0.46) remains high due to hallucinated content.
4. **Hindi_Story_News source**: Stage 2 regresses from Stage 1, suggesting news/literary content doesn't benefit from speaker optimization.
5. **Shekharmeena_TTS source**: Consistently highest error rates across Stage 1 and Stage 2 among all sources.

### Recommendations for Future Training
1. **Increase long-text representation**: Add more 50–200 word samples to fine-tuning data to prevent regression.
2. **Target Very Complex scripts**: Curate training data with literary Hindi, dense conjuncts, and rare grapheme clusters.
3. **Address short-text hallucinations**: Consider architectural or decoding strategy changes (e.g., length-constrained generation) for very short inputs.
4. **Source-specific fine-tuning**: Investigate why Shekharmeena_TTS and Hindi_Story_News show higher error rates — potentially domain-specific vocabulary gaps.

---

## 11. Sample Spotlights

### Best Improvement: HI00334
| Metric | Base | Stage 1 | Stage 2 |
|---|---|---|---|
| WER | 27.3333 | 0.0000 | **0.0000** |

- **Phase**: Phase 3 (Real-World Content) | **Source**: IndicVoices
- The Base Model hallucinated extensively (WER 27.33x), but fine-tuning completely resolved it.

### Worst Regression: HI00373
| Metric | Base | Stage 1 | Stage 2 |
|---|---|---|---|
| WER | 0.2973 | 0.5263 | **1.6316** |

- **Phase**: Phase 3 (Real-World Content) | **Source**: Shekharmeena_TTS
- A well-performing Base Model sample that degraded significantly after fine-tuning. Likely overfitting or domain mismatch in the fine-tuning data.

### Hardest Sample (Stage 2): HI00042
| Metric | Stage 2 |
|---|---|
| WER | 2.0000 |
| CER | 1.9000 |

- The model generates significantly more content than the reference, indicating a persistent hallucination issue on this particular sample.

---

## 12. Visualizations

### Overall Performance

![Orpheus Hindi TTS Overall Performance Across Stages](C:/Users/amanr/.gemini/antigravity-ide/brain/a9cb53ed-4cdd-4187-8503-06fdae93f938/plots/plot_1_overall.png)

### Sample-Level Improvement Distribution

![Sample-Level WER Change Distribution](C:/Users/amanr/.gemini/antigravity-ide/brain/a9cb53ed-4cdd-4187-8503-06fdae93f938/plots/plot_6_improvement_pie.png)

---

## 13. Full Per-Sample Reference Table

> **500 samples** with scores across all three stages. Sorted by Sample ID.

| # | Sample ID | Phase | Source | WER Base | CER Base | WER S1 | CER S1 | WER S2 | CER S2 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | HI00001 | P1 | Rasa_AI4Bharat | 0.0000 | 0.0000 | 1.0000 | 0.2500 | 0.0000 | 0.0000 |
| 2 | HI00002 | P1 | Rasa_AI4Bharat | 0.3889 | 0.1818 | 0.2778 | 0.0758 | 0.1667 | 0.0758 |
| 3 | HI00003 | P1 | Rasa_AI4Bharat | 2.7143 | 2.7206 | 0.2857 | 0.1176 | 0.2143 | 0.0882 |
| 4 | HI00004 | P1 | Rasa_AI4Bharat | 0.1714 | 0.1034 | 0.1714 | 0.1034 | 0.1714 | 0.0690 |
| 5 | HI00005 | P1 | Rasa_AI4Bharat | 0.2000 | 0.0824 | 0.2000 | 0.0471 | 0.2000 | 0.0471 |
| 6 | HI00006 | P1 | Rasa_AI4Bharat | 0.3000 | 0.1613 | 0.3500 | 0.1452 | 0.3500 | 0.1452 |
| 7 | HI00007 | P1 | Rasa_AI4Bharat | 0.1500 | 0.0588 | 0.1000 | 0.0147 | 0.0500 | 0.0441 |
| 8 | HI00008 | P1 | Rasa_AI4Bharat | 1.0000 | 0.9200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 9 | HI00009 | P1 | Rasa_AI4Bharat | 0.2174 | 0.0857 | 0.3043 | 0.1000 | 0.2609 | 0.0857 |
| 10 | HI00010 | P1 | Rasa_AI4Bharat | 0.5385 | 0.3846 | 0.2308 | 0.1538 | 0.3077 | 0.1923 |
| 11 | HI00011 | P1 | Rasa_AI4Bharat | 1.0000 | 0.7500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 12 | HI00012 | P1 | Rasa_AI4Bharat | 0.4375 | 0.3200 | 0.1875 | 0.0800 | 0.3125 | 0.0800 |
| 13 | HI00013 | P1 | Rasa_AI4Bharat | 0.2593 | 0.1429 | 0.2963 | 0.1310 | 0.2222 | 0.0952 |
| 14 | HI00014 | P1 | Rasa_AI4Bharat | 0.5500 | 0.2373 | 0.2500 | 0.0847 | 0.2000 | 0.1017 |
| 15 | HI00015 | P1 | Rasa_AI4Bharat | 0.3333 | 0.1220 | 0.3333 | 0.1220 | 0.1667 | 0.0244 |
| 16 | HI00016 | P1 | Rasa_AI4Bharat | 0.1111 | 0.0541 | 0.1852 | 0.0541 | 0.1481 | 0.0541 |
| 17 | HI00017 | P1 | Rasa_AI4Bharat | 1.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 18 | HI00018 | P1 | Rasa_AI4Bharat | 0.5417 | 0.3585 | 0.3750 | 0.1698 | 0.3750 | 0.1698 |
| 19 | HI00019 | P1 | Rasa_AI4Bharat | 5.7500 | 6.3333 | 0.5000 | 0.0000 | 0.0000 | 0.0000 |
| 20 | HI00020 | P1 | Rasa_AI4Bharat | 0.2692 | 0.0714 | 0.3846 | 0.0714 | 0.3846 | 0.0714 |

> [!NOTE]
> The first 20 samples are shown above for quick reference. The complete 500-sample dataset is available in the companion data file at [analysis_stats.json](file:///c:/Users/amanr/.gemini/antigravity-ide/brain/a9cb53ed-4cdd-4187-8503-06fdae93f938/scratch/analysis_stats.json) under the `per_sample` key.

---

## Appendix: Methodology

### Evaluation Pipeline
1. **Text Input** → Orpheus TTS (Base / Stage 1 / Stage 2) → **Audio Output**
2. **Audio Output** → Whisper-large ASR → **Transcription**
3. **Transcription** vs. **Normalized Reference Text** → WER and CER computation

### Metrics Definition
- **WER (Word Error Rate)**: `(Substitutions + Insertions + Deletions) / Reference Word Count`
- **CER (Character Error Rate)**: `(Substitutions + Insertions + Deletions) / Reference Character Count`

### Normalization
- Text was normalized by removing punctuation, standardizing nukta/chandrabindu variants, and lowercasing where applicable.
- Both reference and ASR transcription underwent identical normalization before comparison.

---

*Report last updated on July 11, 2026. Stage 3 WER/CER metrics added across all three evaluation phases. Data source: 500 Hindi evaluation samples across 3 phases.*
