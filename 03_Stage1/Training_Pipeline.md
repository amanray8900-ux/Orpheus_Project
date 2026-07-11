# Stage 1: LoRA Fine-Tuning Multi-GPU Training Pipeline

This document provides a comprehensive breakdown of the training pipeline used for Stage 1 of the Orpheus TTS model (`kaggle_orpheus_lora_stage1_multigpu.ipynb`). 

> **Mentor Explanation Guide:** You can use this document as a script or reference to explain the architectural choices, memory optimizations, and distributed training setup used to train the model on Kaggle.

---

## 1. Environment & Library Setup
The pipeline runs on a Kaggle environment equipped with dual T4 GPUs. To maximize efficiency:
- We install **Unsloth**, a library highly optimized for faster and more memory-efficient LoRA (Low-Rank Adaptation) fine-tuning.
- We rely on `accelerate` for distributed multi-GPU training, and `bitsandbytes` to enable memory-efficient 8-bit optimizers.
- We authenticate with **Weights & Biases (WandB)** for real-time loss tracking and the **Hugging Face Hub** for seamless checkpoint syncing.

## 2. Dataset Management & Disk Optimization
Handling large audio datasets on Kaggle requires aggressive memory management:
1. **Direct Download:** We pull the dataset manifest and a heavy (7.2GB) `.zip` of audio files directly from Hugging Face.
2. **Extraction & Path Fixing:** The audio is extracted to the local Kaggle working directory. A dynamic mapping function updates the manifest to point to these newly extracted local paths.
3. **Critical Disk Cleanup:** *This is a crucial step to explain.* Immediately after extraction, we delete the 7.2GB zip file and manually clear the Hugging Face cache. Without this step, Kaggle runs out of disk space before training even begins.
4. **Validation Split:** We deterministically carve out a 5% validation set using a fixed seed (42) to ensure consistent evaluation across runs.

## 3. Audio Tokenization (SNAC)
To feed audio into the language model, we must convert continuous audio waveforms into discrete tokens.
- We load the **SNAC (Speech Neural Audio Codec)** model (`hubertsiuzdak/snac_24khz`).
- The audio is resampled to 24kHz.
- SNAC encodes the audio into multiple hierarchical codebooks (different granularities of audio features). 
- **Vocabulary Shifting:** We flatten these multi-level codes into a single 1D sequence. *Crucially, we shift these audio token IDs by a massive offset (e.g., +128266).* This shifts the audio tokens into an unused space in the language model's vocabulary, ensuring the model doesn't confuse a speech token with a text token.
- **Compression:** We apply a function to remove consecutive duplicate acoustic frames, artificially shortening the sequence length without losing semantic meaning. This drastically speeds up attention calculations.

## 4. Input Formatting & Tokenization
The base language model needs a structured way to understand when it is reading text vs. generating speech.
- We define custom special tokens: `<|start_of_human|>`, `<|end_of_human|>`, `<|start_of_ai|>`, `<|start_of_speech|>`, and `<|end_of_speech|>`.
- Every training example is packed into a rigid structure: 
  `[Human] -> Text -> [End Human] -> [AI] -> [Speech] -> Audio Tokens -> [End Speech] -> [End AI]`
- **Context Window Management:** The model has a `max_seq_length` of 2048. We strictly truncate the text, calculate the remaining budget, and truncate the audio tokens (ensuring we respect the 7-token frame boundary of SNAC) so no sample causes a tensor size mismatch error.
- The fully processed dataset is saved to disk, and we aggressively flush the RAM and GPU cache (`gc.collect()`, `torch.cuda.empty_cache()`) to prepare for the heavy training phase.

## 5. Multi-GPU Distributed Training (DDP)
Jupyter notebooks natively struggle with multiprocessing, which is required to train across multiple GPUs simultaneously. To solve this, we generate a standalone python script (`train.py`) from within the notebook.

### The `train.py` Script:
- **Smart Resumption:** The script pings Hugging Face to check if previous checkpoints exist. If it finds one, it downloads it and verifies the integrity of `adapter_model.safetensors`, `optimizer.pt`, and `scheduler.pt`. If verified, it automatically resumes training from that exact step, preventing lost progress if Kaggle disconnects.
- **Training Configuration:**
  - **Optimizer:** `adamw_8bit` (saves VRAM).
  - **Batching:** `per_device_train_batch_size = 2` combined with `gradient_accumulation_steps = 8`. This results in an effective batch size of 16 per GPU (32 total across both GPUs).
  - **Precision:** FP16 mixed-precision is enabled.
  - **Unsloth DDP Fix:** We explicitly set `ddp_find_unused_parameters = False`, which is mathematically required for Unsloth models to train correctly in a distributed setup.

### The Launch Command:
Finally, we trigger the script using:
```bash
!accelerate launch --num_processes 2 train.py
```
This commands the Hugging Face `accelerate` engine to spawn two independent processes. Each process handles a chunk of the batch on its respective T4 GPU, synchronizing gradients via Distributed Data Parallel (DDP) under the hood.
