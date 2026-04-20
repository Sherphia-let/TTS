import os
import json
from tqdm import tqdm
from faster_whisper import WhisperModel

# -----------------------
# CONFIG
# -----------------------
INPUT_DIR = "/data/TTS/sherphia/data/clean_v3/hindi"
OUTPUT_DIR = "/data/TTS/sherphia/transcripted_datas/transcripts_whisper_hindi_clean_v3"

os.makedirs(OUTPUT_DIR, exist_ok=True)

WHISPER_LANG = "hi"

# -----------------------
# LOAD MODEL
# -----------------------
print("Loading Whisper model (large-v3)...")
whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")

# -----------------------
# LOAD FILES (recursive)
# -----------------------
audio_files = []

for root, dirs, files in os.walk(INPUT_DIR):
    for file in files:
        if file.endswith(".wav"):
            audio_files.append(os.path.join(root, file))

audio_files = sorted(audio_files)

print(f"Found {len(audio_files)} audio files...\n")

# -----------------------
# PROCESS FILES
# -----------------------
for path in tqdm(audio_files):

    try:
        segments, info = whisper_model.transcribe(
            path,
            language=WHISPER_LANG,
            beam_size=5   # improves accuracy
        )

        text = " ".join([seg.text for seg in segments]).strip()

    except Exception as e:
        print(f"✘ Failed: {path} | {e}")
        text = ""

    # -----------------------
    # SAVE OUTPUT
    # -----------------------
    rel_path = os.path.relpath(path, INPUT_DIR)
    base_name = os.path.splitext(rel_path)[0]

    output_path = os.path.join(OUTPUT_DIR, base_name + ".json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output_data = {
        "file": path,
        "transcript": text,
        "words": None,
        "confidence": None
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"✔ Saved: {output_path}")

print("\n✅ All files processed!")