# Llama Backbone & Training/Inference Parameters

This document describes the transformer backbone configuration, pretraining data-mixing strategies, and inference sampling parameters of the Orpheus model.

---

## Objective
The objective is to utilize a pre-trained Llama 3B parameter Transformer as the sequential processing engine, configuring hyperparameters for supervised fine-tuning (SFT) and low-latency inference.

## Problem
Standard fine-tuning of 3B parameter models requires significant memory and is slow. Additionally, at inference time, autoregressive speech models are highly sensitive to sampling settings:
1. **Looping Artifacts**: Causal speech models are prone to repeating code combinations, causing infinite audio stutter.
2. **Robotic Speech**: Low sampling temperature causes monotonic speech, while high temperature leads to word hallucination and static noise.

## Hypothesis
Using Unsloth FastLanguageModel and PEFT (LoRA) will allow memory-efficient SFT. Setting strict inference sampling parameters—specifically a high **repetition penalty of 1.3** and a **temperature of 0.6** via vLLM—will suppress looping artifacts and produce natural voice pitch contours.

---

## Method: Training & Inference Configurations

```
    TRAINING FLOW                          INFERENCE FLOW
[ Text & Audio Datasets ]              [ Text Prompt & Voice Tag ]
          │                                        │
          ▼ (Ratio Mixing 2:1)                     ▼ (Formatting)
[ BatchedRatioDataset ]                Sequence: SOH -> Prompt -> SOS
          │                                        │
          ▼ (FSDP / LoRA)                          ▼ (vLLM Inference Engine)
[ 3B Model SFT Updates ]               Parameters: temp=0.6, rep_penalty=1.3
          │                                        │
          ▼                                        ▼
[ Saved Checkpoint ]                   [ Generated Audio Tokens ]
```

### 1. Mixed Pretraining Ratio Strategy
During base pretraining, the model alternates batches of plain text data and TTS data (using a configurable ratio like **2:1 or 1:1**):
* **Text QA Batches**: Preserves semantic, syntactic, and structural understanding of written language.
* **TTS Audio Batches**: Teaches next-token audio projection.
* *Insight*: Alternating batches prevents the model from forgetting language comprehension while learning speech, which directly boosts prosodic naturalness.

### 2. Supervised Fine-Tuning (SFT) Parameters
During target speaker fine-tuning, LoRA adapters are injected:
* **Trainable Matrices**: Targets attention weights (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and MLP layers (`gate_proj`, `up_proj`, `down_proj`).
* **Hyperparameters**: Rank $r=64$ (or $r=32$ for lightweight runs), Alpha $\alpha=64$.
* **Optimizers**: 8-bit AdamW (`adamw_8bit`) and gradient checkpointing (`use_gradient_checkpointing = "unsloth"`) to fit context sequences up to 2048 length in VRAM.

### 3. Inference / Sampling Parameters (vLLM Engine)
For fast, real-time generation, Orpheus utilizes vLLM (`AsyncLLMEngine`) with tuned sampling settings:
* **Temperature** = `0.6`: Controls the randomness of the predictions. Lower values are robotic; higher values generate glitches.
* **Top-p** = `0.8`: Nucleus sampling threshold to filter out low-probability tail tokens.
* **Repetition Penalty** = `1.3` (Critical): Multiplies logits of already generated tokens by a penalty factor, preventing the model from repeating audio frames (preventing looping crashes).
* **Stop Token IDs**: `[49158]` or `[128258]` to stop generation at the end of speech.

---

## Code Implementation

### 1. Training Setup (HuggingFace Trainer & LoRA)
```python
from unsloth import FastLanguageModel
from transformers import TrainingArguments, Trainer

# Initialize model and LoRA
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/orpheus-3b-0.1-ft",
    max_seq_length=2048,
    load_in_4bit=False
)

model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth"
)

# Trainer instantiation
trainer = Trainer(
    model=model,
    train_dataset=dataset,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        output_dir="outputs"
    )
)
```

### 2. Inference Generation Setup (vLLM)
```python
from vllm import AsyncLLMEngine, SamplingParams

# Load inference model
model = AsyncLLMEngine.from_engine_args(engine_args)

# Sampling parameters from report
sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.8,
    repetition_penalty=1.3,      # Prevents audio looping
    max_tokens=1200,
    stop_token_ids=[49158]       # End-of-speech stop token
)
```

---

## Limitations
1. **vLLM Constraints**: High repetition penalties ($>1.5$) can occasionally cause the model to shift voices to bypass penalties.
2. **Context Limits**: 2048 max context caps speech generation to short audio snippets.

## Decision
Employ PEFT LoRA adapters for SFT stages, and deploy the model with the vLLM engine utilizing a repetition penalty of 1.3 and temperature of 0.6.

## Next Stage
Establish the zero-shot **Base Model Evaluation** on the Hindi benchmark dataset.
