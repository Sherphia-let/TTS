# 🎙️ Speech Dataset Creation Pipeline

## 📌 Overview

This project implements a **complete end-to-end pipeline** to convert raw movie/audio data into **high-quality, speaker-labeled speech clips**.
It is designed for use in:

* Text-to-Speech (TTS)
* Automatic Speech Recognition (ASR)
* Speaker Recognition
* Voice AI training datasets

The pipeline ensures **clean, natural, and consistent audio clips** through multiple stages like diarization, denoising, segmentation, and quality filtering.

---

## 🧠 Pipeline Flow

```
Raw Audio
   ↓
Format Normalization
   ↓
Chunking
   ↓
Song Removal
   ↓
Vocal Separation (Demucs)
   ↓
Speaker Diarization
   ↓
Speaker Filtering
   ↓
Natural Splitting
   ↓
Denoising
   ↓
Loudness Normalization
   ↓
Quality Filtering
   ↓
Final Clean Dataset ✅
```

---

## 📁 Project Structure

```
.
├── scripts/
├── data/
│   ├── raw/
│   ├── separated/
│   ├── annotations/
│   ├── final_clips/
│   ├── denoised_clips/
│   ├── normed_clips/
│   ├── clean_clips/
│   └── rejected_clips/
├── .env
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Install Dependencies

```
pip install torch torchaudio soundfile numpy pandas
pip install pyannote.audio python-dotenv
```

### 2. Install FFmpeg

Required for audio processing:

```
ffmpeg -version
```

### 3. Setup Hugging Face Token

Create `.env` file:

```
HF_token=your_token_here
```

---

## 🚀 Script Descriptions

### 🔹 `format_normalize.py`

Converts all input audio into a standardized format (mono, 24kHz, 16-bit).
This ensures consistency across the pipeline and avoids compatibility issues in later stages.
Acts as the foundational preprocessing step before any splitting or modeling.

---

### 🔹 `splits_sudio.py`

Splits long audio files into smaller chunks (typically ~5 minutes).
This improves processing efficiency and prevents memory issues during diarization and separation.
Also helps parallelize later stages of the pipeline.

---

### 🔹 `remove_songs.py`

Detects and removes music-heavy segments based on energy and acoustic patterns.
Ensures that only dialogue/speech portions are retained for dataset creation.
Improves overall dataset quality by eliminating non-speech noise.

---

### 🔹 `demucs_separate.py`

Uses **Demucs** to separate vocals from background sounds.
Extracts clean speech signals while suppressing music and noise.
A critical step for improving diarization and final audio quality.

---

### 🔹 `diarize_and_split.py`

Performs speaker diarization using **Pyannote**.
Identifies “who spoke when” and generates timestamped segments.
Outputs annotations (RTTM/JSON) used for further speaker-based processing.

---

### 🔹 `normalize_speaker.py`

Standardizes speaker labels into a consistent format (e.g., SPEAKER_01).
Removes inconsistencies across different chunks or files.
Ensures uniform labeling across the entire dataset.

---

### 🔹 `rttm_speaker_duration.py`

Calculates total speaking duration for each speaker.
Filters out speakers with insufficient data (low duration).
Helps retain only meaningful and usable speaker samples.

---

### 🔹 `natural_split.py`

Splits speech into natural, human-like segments (2–8 seconds).
Avoids abrupt cuts, overlaps, and duplicate segments.
Produces clean clips suitable for TTS and training models.

---

### 🔹 `denoise.py`

Applies denoising using **DeepFilterNet**.
Removes background noise and enhances speech clarity.
Requires a separate environment for optimized performance.

---

### 🔹 `loudnorm.py`

Normalizes audio loudness to a consistent level (LUFS standard).
Ensures all clips have uniform volume and listening quality.
Important for professional-grade datasets.

---

### 🔹 `quality_filter.py`

Final filtering stage using multiple checks:

* Signal-to-noise ratio (SNR)
* Voice Activity Detection (VAD)
* Clipping detection
* Duration limits

### 🔹 `transcribe.py`

Performs Automatic Speech Recognition (ASR) to convert speech into text.

- **Primary model:** Omnilingual
- **Secondary (fallback):** Whisper `large-v3` (used when primary confidence is low)

Features:
- Supports multilingual transcription (Tamil, Hindi, etc.)
- Uses beam search for better accuracy
- Generates word-level timestamps

Fallback logic:
- If primary model confidence < threshold → fallback to Whisper `large-v3`

Outputs per clip:
- Transcript
- Word timestamps
- Confidence score

👉 This stage converts **speech → raw text**

### 🔹 `gemma_transcript_qa.py`

Uses **Gemma 4B (LLM)** for transcript quality assurance and correction.

Performs 3 key tasks:

1. **ASR Error Correction**
   - Fixes obvious transcription mistakes  
   - Improves linguistic accuracy  

2. **Code-Mixing Detection**
   - Detects English words inside Tamil/Hindi  
   - Marks them as:
     ```
     [EN]word[/EN]
     ```

3. **Hallucination Detection**
   - Flags transcripts that are too long for the audio duration  
   - Prevents fake/incorrect text generation  

Filtering:
- Reject clips with confidence < 0.6  

Outputs:
- Corrected transcript  
- Confidence score  
- Codemix flag  
- Reject flag  

👉 This stage ensures **clean, reliable text aligned with audio**

---

### 🔹 `text_normalize.py`

Converts raw transcripts into **TTS-ready normalized text**.

Handles:

- Numbers → spoken form  
  - `1,23,456` → “one lakh twenty three thousand…”  
- Dates → spoken format  
- Currency → readable format  
- Abbreviations → expanded form  
- Phone numbers → digit-wise speech  
- URLs / symbols → cleaned  

Hybrid approach:
- Rule-based normalization (fast + consistent)  
- Gemma 4B for complex/ambiguous cases  

Outputs:
- `raw_text` (original ASR)
- `normalized_text` (final TTS input)

👉 This stage ensures **text is natural, readable, and model-ready**

Separates clips into:

* ✅ Clean dataset
* ❌ Rejected clips (with logs)

---
## 🧪 Environment Architecture

This project uses 3 isolated environments:

### 1️⃣ ASR Environment (Omnilingual / Whisper)
- Speech → text conversion
- File: `requirements_omni_asr.txt`

### 2️⃣ Demucs Environment
- Audio cleaning, separation, diarization
- File: `requirements_demucs.txt`

### 3️⃣ Gemma Environment (vLLM)
- Transcript QA + text normalization
- File: `requirements_gemma.txt`

---
## ⚠️ Important Notes

- vLLM requires CUDA 12.9 nightly build
- DeepGEMM is disabled:
  export VLLM_USE_DEEP_GEMM=0
- Each stage runs in a separate virtual environment to avoid dependency conflicts

---
## 📊 Final Output

```
data/clean_v3/
```

Contains:

* High-quality speech clips
* Speaker-wise organized data
* Ready for ML training
