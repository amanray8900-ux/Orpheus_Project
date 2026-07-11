# Stage 1: Data Preprocessing Pipeline

This document explains the preprocessing pipeline (`hindi_colab_pipeline_rasa_updated.ipynb`) utilized to clean and prepare the training dataset for Stage 1 of the Orpheus TTS project. The pipeline operates on the Rasa HuggingFace dataset and applies rigorous text and audio filtering to ensure high-quality training data.

## 1. Text Normalization and Phonemization

- **Text Normalization**: The raw Hindi text is passed through the `HindiNormalizer` to handle edge cases, expand numbers, abbreviations, dates, times, and URLs into spoken words, and remove extraneous punctuation.
- **Phonemization**: The normalized text is converted into a sequence of phonemes using `espeak-ng`. This provides the phonetic ground truth needed to train pronunciation models.

## 2. Audio Processing and Filtering

To guarantee the quality of the training audio, each sample undergoes several DSP (Digital Signal Processing) steps:

- **Resampling**: All audio is converted to a uniform 16kHz sampling rate.
- **Voice Activity Detection (VAD)**: The `silero-vad` model is used to detect and isolate speech. Non-speech sections (silences) longer than 500ms are trimmed out to prevent the model from learning to synthesize dead air.
- **Duration Filtering**: Audio segments shorter than 2.0 seconds or longer than 18.0 seconds are discarded to maintain a consistent sequence length during training.
- **Clipping Detection**: The pipeline checks for audio clipping (distortion caused by excessively high volume). If more than 1% of the audio samples are clipped (amplitude >= 0.95), the file is rejected.
- **Signal-to-Noise Ratio (SNR)**: Background noise is evaluated by calculating the SNR. Audio samples with an SNR below 8.0 dB are rejected to ensure clean, noiseless training data.
- **Loudness Normalization**: Using the `pyloudnorm` library, audio files are standardized to a target loudness of -23.0 LUFS (Loudness Units relative to Full Scale), ensuring volume consistency across the dataset.

## 3. Manifest Generation

For every audio file that passes the quality gates, the pipeline saves the normalized `.wav` file and extracts rich metadata to build a dataset manifest. 

The output includes:
- **`final_manifest.jsonl` / `final_manifest.parquet`**: A structured database containing:
  - File ID and Source
  - Raw Text, Normalized Text, and extracted Phonemes
  - Original and Processed Durations
  - Calculated SNR, Clipping Fraction, and Loudness (LUFS)
  - Speaker Metadata (Gender, Age Group, State/District)
- **Quality Report**: A generated text report (`quality_report.txt`) summarizing the overall yield (total passed vs. rejected items) and detailing the rejection breakdown (e.g., how many failed VAD, SNR, clipping, or duration checks) along with dataset distributions.

This extensive preprocessing guarantees that Stage 1 training is fed with clean, consistent, and well-documented data.
