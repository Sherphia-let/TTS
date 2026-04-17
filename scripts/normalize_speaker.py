import os
import json

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_JSON = os.path.join(BASE_DIR, "data/annotations/segments.jsonl")
INPUT_RTTM = os.path.join(BASE_DIR, "data/annotations/vocals.rttm")

OUT_JSON = os.path.join(BASE_DIR, "data/annotations/normalized_segments.jsonl")
OUT_RTTM = os.path.join(BASE_DIR, "data/annotations/normalized.rttm")

# =========================
# STEP 1: READ JSON
# =========================
segments = []

with open(INPUT_JSON, "r") as f:
    for line in f:
        segments.append(json.loads(line))

# Sort by start time
segments = sorted(segments, key=lambda x: x["start"])

# =========================
# STEP 2: CREATE MAPPING
# =========================
speaker_map = {}
speaker_count = 1

for seg in segments:
    spk = seg["speaker_id"]

    if spk not in speaker_map:
        speaker_map[spk] = f"SPEAKER_{speaker_count:02d}"
        speaker_count += 1

print("🔁 Speaker Mapping:")
for k, v in speaker_map.items():
    print(f"{k} → {v}")

# =========================
# STEP 3: WRITE NEW JSON
# =========================
with open(OUT_JSON, "w") as f:
    for seg in segments:
        new_seg = seg.copy()
        new_seg["speaker_id"] = speaker_map[seg["speaker_id"]]
        f.write(json.dumps(new_seg) + "\n")

# =========================
# STEP 4: UPDATE RTTM
# =========================
with open(INPUT_RTTM, "r") as f:
    lines = f.readlines()

new_lines = []

for line in lines:
    parts = line.strip().split()

    old_speaker = parts[7]
    new_speaker = speaker_map.get(old_speaker, old_speaker)

    parts[7] = new_speaker
    new_lines.append(" ".join(parts))

with open(OUT_RTTM, "w") as f:
    for line in new_lines:
        f.write(line + "\n")

print("\n✅ Speaker normalization complete!")
print(f"📄 JSON: {OUT_JSON}")
print(f"📄 RTTM: {OUT_RTTM}")