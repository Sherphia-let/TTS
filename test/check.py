import os
import json
from tqdm import tqdm
import torch
import torch.nn.functional as F

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

# -----------------------
# CONFIG
# -----------------------
MODEL_NAME = "omniASR_CTC_7B_v2"

INPUT_DIR = "/data/TTS/sherphia/data/raw/movies/Tamil/iisc_mile"
OUTPUT_DIR = "/data/TTS/sherphia/data/transcripts_ctc_iisc_mile"

os.makedirs(OUTPUT_DIR, exist_ok=True)

LANG_CODE = "tam_Taml"
BATCH_SIZE = 4

# -----------------------
# LOAD MODEL
# -----------------------
print(f"Loading model: {MODEL_NAME}...")
pipeline = ASRInferencePipeline(model_card=MODEL_NAME)

# -----------------------
# CONFIDENCE FUNCTION
# -----------------------
def estimate_confidence_from_text(text):
    # fallback if logits not accessible
    if not text:
        return 0.0
    words = text.split()
    return round(min(1.0, len(words) / 10), 4)

# -----------------------
# SORT FILES
# -----------------------
def natural_sort_key(filename):
    return int(os.path.splitext(filename)[0])

audio_files = sorted(
    [f for f in os.listdir(INPUT_DIR) if f.endswith(".wav")],
    key=natural_sort_key
)

# -----------------------
# PROCESS
# -----------------------
for i in tqdm(range(0, len(audio_files), BATCH_SIZE)):
    batch_files = audio_files[i:i + BATCH_SIZE]

    audio_paths = [os.path.join(INPUT_DIR, f) for f in batch_files]
    lang_list = [LANG_CODE] * len(audio_paths)

    try:
        outputs = pipeline.transcribe(
            audio_paths,
            lang=lang_list,
            batch_size=len(audio_paths)
        )

        for file_name, text in zip(batch_files, outputs):

            # ⚠️ No logits access → fallback confidence
            confidence = estimate_confidence_from_text(text)

            output_data = {
                "file": file_name,
                "transcript": text.strip(),
                "words": None,
                "confidence": confidence
            }

            base_name = os.path.splitext(file_name)[0]
            output_path = os.path.join(OUTPUT_DIR, base_name + ".json")

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            print(f"✔ Saved: {output_path}")

    except Exception as e:
        print(f"✘ Error: {e}")

print("\n✅ Done!")