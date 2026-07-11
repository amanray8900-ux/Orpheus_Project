# SNAC Neural Audio Codec Integration & Streaming Flow

This document describes the multi-scale token interleaving pattern, token-to-code mapping math, and the sliding-window decoding flow of the SNAC codec.

---

## Objective
The objective is to integrate the multi-scale SNAC neural audio codec into the Orpheus TTS pipeline to compress continuous 24kHz waveforms into discrete codes, and to reconstruct raw waveforms from generated token sequences.

## Problem
Text transformers process discrete, one-dimensional sequences. Audio waveforms are continuous and contain 24,000 samples per second. Directly generating waveforms leads to huge sequences and regression averaging artifacts, while simple single-rate codecs either lose high-frequency detail or require too many tokens per second.

## Hypothesis
Using a multi-scale neural codec like SNAC allows compressing audio into three hierarchical layers at different temporal rates (12Hz, 24Hz, 48Hz). Flattening these into an interleaved pattern of 7 tokens per audio frame preserves global prosody and fine phonetic textures. A sliding-window decoder with overlap-add windowing can reconstruct high-fidelity waveforms from these tokens with very low latency (~200ms).

---

## Method: Tokenization, Mapping, & Decoding Flow

```
[ Raw Audio Waveform (24kHz) ]
              │
              ▼ (SNAC Encode)
Codes: Level 1 (12Hz), Level 2 (24Hz), Level 3 (48Hz)
              │
              ▼ (Interleave & Apply Offsets)
Unified Causal Token Sequence: 7 tokens per frame
              │
              ▼ (Inference Generation via vLLM)
Generated Audio Token Stream (Buffer of 28 tokens / 4 frames)
              │
              ▼ (Decoder Sliding Window)
Turn Token ID to SNAC Code (position % 7 math)
              │
              ▼ (SNAC.decode)
Reconstructed Waveform Chunks (Overlap-Add [2048:4096] windowing)
              │
              ▼
[ int16 Audio Waveform Output ]
```

### 1. Multi-Scale Codebooks
SNAC (`hubertsiuzdak/snac_24khz`) compresses audio into three layers:
* **Level 1 (Coarse/Prosody)**: 12Hz rate (1 code per frame). Captures broad spectral contours.
* **Level 2 (Medium)**: 24Hz rate (2 codes per frame). Captures midrange formant transitions.
* **Level 3 (Fine)**: 48Hz rate (4 codes per frame). Captures fine voice details and consonants.

### 2. 7-Token Frame Interleaving Pattern
To combine these rates, codes are flattened into 7-token frames (representing 1/12th of a second):
$$\text{Frame } i = [Q_0(i), Q_1(2i), Q_2(4i), Q_2(4i+1), Q_1(2i+1), Q_2(4i+2), Q_2(4i+3)]$$
Where $Q_0$ is Level 1, $Q_1$ is Level 2, and $Q_2$ is Level 3.

### 3. Token-to-Code Vocabulary Offsets
To keep audio tokens separate from standard text, offsets are applied. Depending on the pipeline layer, two offset mappings are used:

* **Tokenizer Vocabulary ID Mapping (HuggingFace/Unsloth)**:
  Base offset `128,266` (Tokenizer length `128,256` + 10 special tokens):
  $$\text{Vocab ID} = \text{SNAC Code} + 128266 + (\text{position} \pmod 7 \times 4096)$$
* **Custom Token Index Mapping (vLLM output string parsing)**:
  LLM outputs custom token strings like `"<custom_token_4523>"`. The index `number` (e.g. 4523) maps to SNAC codes:
  $$\text{SNAC Code} = \text{number} - 10 - (\text{position} \pmod 7 \times 4096)$$

### 4. Sliding Window & Overlap-Add Streaming Flow
To stream audio before the entire sentence is generated:
1. **Token Buffer**: Incoming tokens are collected in a buffer.
2. **Buffer Threshold**: Once the buffer holds at least **28 tokens (4 frames)**, a decode step is triggered.
3. **Decoupled Reconstruction**: The 28 tokens are split into their respective 3 codebook arrays and passed to `SNAC.decode()`.
4. **Overlap-Add Windowing**: To prevent cracking and stitching artifacts at chunk boundaries, only samples `[2048:4096]` are extracted from the decoded waveform, converted to int16, and yielded.
5. **Yield rate**: Each frame (7 tokens) represents ~7ms of audio; 4 frames of decoded output (after overlap-add windowing) yield **~85ms of audio per chunk**.

---

## Code Implementation

### Audio Tokenization & Offsets (Preprocessing)
```python
def tokenise_audio(waveform, ds_sample_rate):
    # Resample to 24kHz required by SNAC
    waveform = torch.from_numpy(waveform).unsqueeze(0).to(dtype=torch.float32)
    resample_transform = T.Resample(orig_freq=ds_sample_rate, new_freq=24000)
    waveform = resample_transform(waveform).unsqueeze(0).to("cuda")

    # Encode with SNAC
    with torch.inference_mode():
        codes = snac_model.encode(waveform)

    # Interleave and apply base offset 128266 (tokenizer index)
    all_codes = []
    for i in range(codes[0].shape[1]):
        all_codes.append(codes[0][0][i].item() + 128266)
        all_codes.append(codes[1][0][2*i].item() + 128266 + 4096)
        all_codes.append(codes[2][0][4*i].item() + 128266 + (2 * 4096))
        all_codes.append(codes[2][0][(4*i)+1].item() + 128266 + (3 * 4096))
        all_codes.append(codes[1][0][(2*i)+1].item() + 128266 + (4 * 4096))
        all_codes.append(codes[2][0][(4*i)+2].item() + 128266 + (5 * 4096))
        all_codes.append(codes[2][0][(4*i)+3].item() + 128266 + (6 * 4096))
    return all_codes
```

### Sliding-Window Token-to-Audio Decoding (Inference)
```python
def turn_token_into_id(token_string, position):
    # Extracts index N from "<custom_token_N>"
    number = int(token_string.split("_")[-1].replace(">", ""))
    # Subtract custom token offset (10) and codebook base limits
    snac_code = number - 10 - ((position % 7) * 4096)
    return snac_code

def convert_to_audio(multiframe_tokens):
    # Takes 28 tokens (4 frames)
    layer_1 = []
    layer_2 = []
    layer_3 = []
    
    for i in range(4):
        pos = 7 * i
        layer_1.append(multiframe_tokens[pos])
        layer_2.append(multiframe_tokens[pos+1] - 4096)
        layer_3.append(multiframe_tokens[pos+2] - (2 * 4096))
        layer_3.append(multiframe_tokens[pos+3] - (3 * 4096))
        layer_2.append(multiframe_tokens[pos+4] - (4 * 4096))
        layer_3.append(multiframe_tokens[pos+5] - (5 * 4096))
        layer_3.append(multiframe_tokens[pos+6] - (6 * 4096))
        
    codes = [
        torch.tensor(layer_1).unsqueeze(0).to("cuda"),
        torch.tensor(layer_2).unsqueeze(0).to("cuda"),
        torch.tensor(layer_3).unsqueeze(0).to("cuda")
    ]
    
    with torch.inference_mode():
        audio_hat = snac_model.decode(codes)
        
    # Overlap-add slice to avoid boundary clicks
    audio_chunk = audio_hat[:, :, 2048:4096].squeeze().cpu().numpy()
    # Convert float32 to 16-bit PCM bytes
    audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
    return audio_bytes
```

---

## Limitations
1. **Overlap Overhead**: Slicing windowed frames throws away half the generated samples, doubling decoder workload.
2. **Out-of-Range Codes**: If the LLM generates a token representing a code $>4096$, decoding will fail.

## Decision
Utilize the 7-token interleaved mapping with custom offset decoding and a 28-token sliding-window buffer for real-time streaming inference.

## Next Stage
Integrate the Llama transformer backbone model and fine-tuning parameters.
