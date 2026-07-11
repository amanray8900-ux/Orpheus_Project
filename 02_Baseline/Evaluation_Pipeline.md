# ASR and Evaluation Metrics Pipeline

The ASR and Evaluation Metrics pipeline (`ASR_code.ipynb`) automates the process of transcribing synthesized audio and computing standard speech recognition metrics against the ground truth dataset. It leverages OpenAI's Whisper model to evaluate the performance of the Orpheus TTS system.

## Pipeline Workflow

1. **Data Acquisition via Google Drive:**
   - The notebook integrates with Google Drive using the `pydrive2` library.
   - It searches for and automatically downloads the generated `.wav` audio files (e.g., from the `Re-Phase1_after normalisation` folder).
   - It also retrieves the corresponding dataset ground truth JSON file.

2. **ASR Model Initialization:**
   - The pipeline utilizes **OpenAI's Whisper**, specifically the `large` model variant (1.5B parameters).
   - The `large` model is selected because it provides the highest accuracy for Hindi language transcription.
   - The model is loaded onto a GPU (CUDA) to ensure the transcription process runs efficiently.

3. **Transcription Process:**
   - The pipeline iterates through all downloaded audio files and transcribes them.
   - The transcription is constrained to Hindi (`language='hi'`) and is performed strictly as a transcription task (`task='transcribe'`), meaning no English translation is applied.
   - The resulting transcriptions are saved locally into a JSON file (`whisper_transcriptions_re-phase1_after normalisation.json`) and subsequently uploaded back to Google Drive for persistence.

4. **Normalization & Metric Calculation:**
   - Although the notebook's full evaluation code is truncated in this overview, it relies heavily on the `jiwer` library.
   - Both the generated Whisper transcriptions and the original ground truth dataset text are passed through the `HindiNormalizer`. This strips formatting and ensures that metrics aren't skewed by trivial differences (e.g., Arabic vs. Devanagari numerals, punctuation, abbreviation expansions).
   - The pipeline calculates two primary evaluation metrics:
     - **WER (Word Error Rate):** Evaluates accuracy on a per-word basis.
     - **CER (Character Error Rate):** Evaluates accuracy on a per-character basis, which is highly relevant for agglutinative and phonetically complex languages like Hindi.
