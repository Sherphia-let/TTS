import os
import json
from tqdm import tqdm

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
from faster_whisper import WhisperModel

# -----------------------
# CONFIG
# -----------------------
MODEL_NAME = "omniASR_LLM_3B_v2"
# MODEL_NAME = "omniASR_LLM_Unlimited_3B_v2"  # use this if audio > 40 sec

INPUT_DIR = "/data/TTS/sherphia/data/raw/Tamil/spring_inx_r2"
OUTPUT_DIR = "/data/TTS/sherphia/test/transcripts_omnilingual_spring_inx_r2"

os.makedirs(OUTPUT_DIR, exist_ok=True)

LANG_CODE = "tam_Taml"  # Tamil
WHISPER_LANG = "ta"

BATCH_SIZE = 4  # adjust based on GPU memory

# -----------------------
# LOAD MODELS
# -----------------------
print(f"Loading Omnilingual ASR model: {MODEL_NAME}...")
pipeline = ASRInferencePipeline(model_card=MODEL_NAME)

print("Loading Whisper fallback model (large-v3)...")
whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")

# -----------------------
# SORT FILES
# -----------------------
def natural_sort_key(filename):
    return int(os.path.splitext(filename)[0])

audio_files = sorted(
    [f for f in os.listdir(INPUT_DIR) if f.endswith(".wav")],
    key=natural_sort_key
)

print(f"Found {len(audio_files)} audio files...\n")

# -----------------------
# PROCESS IN BATCHES
# -----------------------
for i in tqdm(range(0, len(audio_files), BATCH_SIZE)):
    batch_files = audio_files[i:i + BATCH_SIZE]

    audio_paths = [os.path.join(INPUT_DIR, f) for f in batch_files]
    lang_list = [LANG_CODE] * len(audio_paths)

    try:
        # =======================
        # OMNILINGUAL TRANSCRIBE
        # =======================
        transcriptions = pipeline.transcribe(
            audio_paths,
            lang=lang_list,
            batch_size=len(audio_paths)
        )

    except Exception as e:
        print(f"⚠ Omnilingual failed for batch starting at {batch_files[0]}: {e}")
        print("🔁 Falling back to Whisper...\n")

        transcriptions = []

        # =======================
        # WHISPER FALLBACK
        # =======================
        for path in audio_paths:
            try:
                segments, _ = whisper_model.transcribe(
                    path,
                    language=WHISPER_LANG
                )
                text = " ".join([seg.text for seg in segments])
                transcriptions.append(text.strip())

            except Exception as we:
                print(f"✘ Whisper also failed for {path}: {we}")
                transcriptions.append("")

    # =======================
    # SAVE OUTPUT
    # =======================
    for file_name, text in zip(batch_files, transcriptions):

        output_data = {
            "file": file_name,
            "transcript": text.strip(),
            "words": None,
            "confidence": None
        }

        base_name = os.path.splitext(file_name)[0]
        output_path = os.path.join(OUTPUT_DIR, base_name + ".json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"✔ Saved: {output_path}")

print("\n✅ All files processed!")