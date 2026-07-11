# Orpheus TTS Hindi Project Documentation

Welcome to the comprehensive documentation repository for the **Orpheus TTS Hindi (3B)** speech synthesis model. This project covers the journey of adapting, training, and optimizing the Orpheus autoregressive speech generation model specifically for Hindi and Hinglish speech synthesis with high phonetic accuracy, speaker consistency, and emotional expression.

## Directory Structure

This documentation is organized sequentially, following the phases of the project:

```
Orpheus_Project/
├── README.md                          # Document index and directory map
├── 00_Project_Overview.md              # Slide-friendly high-level overview
├── 01_Background/
│   ├── Orpheus_Architecture.md        # Core autoregressive TTS concepts
│   ├── SNAC.md                        # Neural audio codec details
│   └── Llama.md                       # Backbone LLM and PEFT parameters
├── 02_Baseline/
│   └── Base_Model_Evaluation.md       # Initial evaluation of the base model
├── 03_Stage1/
│   └── Pronunciation.md               # Hindi pronunciation fine-tuning & failure analysis
├── 04_Stage2/
│   └── Speaker_Identity.md            # Speaker-conditioned voice learning
├── 05_Stage3/
│   └── Emotion.md                     # Joint speaker-emotion conditioning
└── 06_Future_Work/
    ├── GRPO.md                        # RL optimization with Group Relative Policy Optimization
    ├── LLM_Replacement.md             # Moving to Gemma-3-270M or Qwen3-4B backbones
    └── Emotion_Analysis.md            # Richness and speaker consistency metrics
└── 07_Results/
    ├── Metrics.md                     # Tabulated evaluation results (WER/CER)
    └── AudioExamples.md               # Guide for evaluating audio outputs
```

---

## Documentation Sections (Standard Layout)

For each stage and background component, the documentation details:
- **Objective**: The goal of this specific phase.
- **Problem**: The technical challenges faced.
- **Hypothesis**: The proposed solution or design.
- **Dataset**: Data sources, preprocessing, and filtering parameters.
- **Method**: The algorithm, pipeline, and architecture adjustments.
- **Training Configuration**: Hyperparameters, hardware details, and optimization settings.
- **Evaluation**: Verification metrics and methodology (e.g., Whisper ASR, jiwer WER/CER).
- **Results**: Performance improvements and milestones.
- **Limitations**: Remaining issues and design gaps.
- **Decision**: Tactical and strategic choices made based on outcomes.
- **Next Stage**: The subsequent step in the project.
