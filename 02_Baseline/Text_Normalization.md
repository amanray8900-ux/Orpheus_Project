# Text Normalization Pipeline

The text normalization pipeline (`text_normalizer.py`) is designed to preprocess and standardize Hindi text prior to synthesis by the TTS system, as well as for calculating accurate evaluation metrics (WER and CER). It handles various edge cases, formatting anomalies, and domain-specific expansions.

## Core Component: `HindiNormalizer`

The `HindiNormalizer` class provides the following key text processing capabilities:

1. **Formatting and Punctuation Cleanup:** 
   - Unescapes HTML entities.
   - Strips out zero-width characters (`\ufeff`, `\u200b`) and bullet markers.
   - Removes standard punctuation marks to yield clean, space-separated words.
   
2. **Number & Digit Expansion:** 
   - Translates Hindi/Devanagari numerals (`०-९`) to standard Arabic numerals (`0-9`).
   - Converts numeric values into spoken Hindi words (e.g., expanding decimals and large integers using the `indic-numtowords` library).

3. **Latin Character and Abbreviation Handling:**
   - Expands individual Latin characters and well-known English abbreviations (e.g., `ISRO` -> `इसरो`, `WHO` -> `डब्लूएचओ`) into their Hindi pronunciations.
   - Maps standard Hindi abbreviations containing dots to their full spoken forms (e.g., `डॉ.` -> `डॉक्टर`, `कि.मी.` -> `किलोमीटर`).

4. **URL and Email Expansion:**
   - Detects and translates email addresses and URLs into spoken format, translating characters like `@` to `एट`, `.` to `डॉट`, and expanding common domains (`gmail` -> `जीमेल`, `com` -> `कॉम`).

5. **Social Media Entities:**
   - Extracts the core text from hashtags (`#`) and expands user mentions (`@handle`) digit-by-digit for natural reading.

6. **Phone Numbers & OTPs:**
   - Intelligently recognizes digits based on contextual keywords (like "OTP", "पिन", "कोड"). When matched, it expands the digits to be read individually rather than as a whole number. Also handles 10-digit Indian phone numbers (with or without the `+91` prefix).

7. **Units & Currencies:**
   - Expands measurement units (e.g., `°C` -> `डिग्री सेल्सियस`, `GB` -> `गीगाबाइट`) and currency symbols (`₹`, `$`, `€`, `£`) into full spoken Hindi.

8. **Dates & Times:**
   - Formats standard date strings (`DD/MM/YYYY`) into spoken format (e.g., `1 जनवरी 2024`).
   - Translates time formats (`HH:MM`) into spoken strings like `4 बजकर 30 मिनट`.

9. **Math Operators & Percentages:**
   - Converts mathematical symbols (`+`, `-`, `=`, `×`, `÷`) and superscripts (`²`, `³`) into their Hindi equivalents (`जमा`, `घटा`, `बराबर`, `गुणा`, `भाग`, `वर्ग`, `घन`).
   - Translates the `%` symbol to `प्रतिशत`.

10. **Phonetic and Script Corrections:**
    - Corrects conjunct anusvara combinations to maintain phonetic consistency.
    - Explicitly handles Devanagari Nukta compositions (e.g., mapping `क़` to `क` and handling decomposed Unicode variants) to standardize Urdu-origin words and remove stray Nukta modifiers.

By uniformly applying this normalization to both the reference ground truth text and the transcription outputs, the pipeline ensures a fair and accurate, "apples-to-apples" comparison during model evaluation.
