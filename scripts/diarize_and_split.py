import os
import torch
import soundfile as sf
import json
from pyannote.audio import Pipeline
from dotenv import load_dotenv

# =========================
# CONFIG
# =========================

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"))

HF_TOKEN = os.getenv("HF_token")


if HF_TOKEN is None:
    raise ValueError("HF_TOKEN not found")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_dir = os.path.join(BASE_DIR, "data/separated/movies_clean")
annotations_dir = os.path.join(BASE_DIR, "data/annotations")

os.makedirs(annotations_dir, exist_ok=True)

# =========================
# LOAD MODEL
# =========================
print("🧠 Loading diarization model...")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=HF_TOKEN
)

if torch.cuda.is_available():
    pipeline.to(torch.device("cuda"))
    print("🚀 Using GPU")
else:
    print("💻 Using CPU")

# =========================
# PROCESS FILES
# =========================
jsonl_path = os.path.join(annotations_dir, "segments.jsonl")

with open(jsonl_path, "w", encoding="utf-8") as jsonl_file:

    for file in os.listdir(input_dir):
        if not file.endswith(".wav"):
            continue

        input_path = os.path.join(input_dir, file)
        file_name = os.path.splitext(file)[0]

        print(f"\n🎙️ Processing: {file}")

        # Load audio safely
        audio, sr = sf.read(input_path)

        waveform = torch.tensor(audio)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.T

        waveform = waveform.float()

        # =========================
        # DIARIZATION (NO SPLITTING)
        # =========================
        diarization = pipeline({
            "waveform": waveform,
            "sample_rate": sr
        })

        # =========================
        # SAVE RTTM
        # =========================
        rttm_path = os.path.join(annotations_dir, f"{file_name}.rttm")

        with open(rttm_path, "w") as f:
            diarization.speaker_diarization.write_rttm(f)

        print(f"📄 RTTM saved: {rttm_path}")

        # =========================
        # SAVE JSONL
        # =========================
        for segment, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):

            duration = segment.end - segment.start

            # requirement: 2–20 sec
            if duration < 2 or duration > 20:
                continue

            json_entry = {
                "file": file_name,
                "speaker_id": f"{file_name}_{speaker}",
                "start": round(segment.start, 2),
                "end": round(segment.end, 2)
            }

            jsonl_file.write(json.dumps(json_entry) + "\n")

print("\n🎯 Diarization complete (NO audio splitting)")