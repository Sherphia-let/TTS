import os
import json
import hashlib
import numpy as np
import soundfile as sf
from collections import defaultdict

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AUDIO_PATH = os.path.join(BASE_DIR, "data/separated/movies_clean/vocals.wav")
RTTM_PATH = os.path.join(BASE_DIR, "data/annotations/vocals.rttm")
VALID_SPK_PATH = os.path.join(BASE_DIR, "data/annotations/valid_speakers.json")

OUTPUT_DIR = os.path.join(BASE_DIR, "data/final_clips")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# CONFIG
# =========================
MIN_DURATION = 2.2
MAX_DURATION = 8.0
TARGET_MIN = 3.0
TARGET_MAX = 6.0

MERGE_GAP = 0.55 # merge segments if gap < 0.5 sec
seen_hashes = set() # to track duplicates
print("NEW FILE LOADING")

rejected_overlap = 0
rejected_duplicate = 0

# =========================
# LOAD VALID SPEAKERS
# =========================
with open(VALID_SPK_PATH, "r") as f:
    valid_speakers = set(json.load(f))

print(f"✅ Valid speakers loaded: {len(valid_speakers)}")

# =========================
# LOAD AUDIO
# =========================
audio, sr = sf.read(AUDIO_PATH)

if audio.ndim == 2:
    audio = np.mean(audio, axis=1)

def is_single_speaker(start, end, speaker, segments):
    for seg in segments:
        if seg["speaker"] == speaker:
            continue

        # check overlap
        if not (end <= seg["start"] or start >= seg["end"]):
            return False
    return True


# =========================
# PARSE RTTM
# =========================
def parse_rttm():
    segments = []

    with open(RTTM_PATH, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue

            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]

            if speaker not in valid_speakers:
                continue

            segments.append({
                "start": start,
                "end": start + duration,
                "speaker": speaker
            })

    return sorted(segments, key=lambda x: x["start"])

segments = parse_rttm()
print(f"🎙️ Loaded segments: {len(segments)}")

# =========================
# REMOVE OVERLAPS
# =========================
clean_segments = []

for i in range(len(segments)):
    curr = segments[i]

    overlap = False
    for j in range(len(segments)):
        if i == j:
            continue

        other = segments[j]

        if not (curr["end"] <= other["start"] or curr["start"] >= other["end"]):
            if curr["speaker"] != other["speaker"]:
                overlap = True
                break

    if not overlap:
        clean_segments.append(curr)

print(f"🧹 Non-overlapping segments: {len(clean_segments)}")

# =========================
# MERGE CLOSE SEGMENTS (same speaker)
# =========================
merged = []
prev = clean_segments[0]

for curr in clean_segments[1:]:
    if (
        curr["speaker"] == prev["speaker"] and
        curr["start"] - prev["end"] < MERGE_GAP
    ):
        prev["end"] = curr["end"]
    else:
        merged.append(prev)
        prev = curr

merged.append(prev)

print(f"🔗 Merged segments: {len(merged)}")

# =========================
# SPLIT INTO 2–8 SEC CLIPS
# =========================
clip_id = 0

def save_clip(start, end, speaker):
    global clip_id, rejected_overlap, rejected_duplicate

    duration = end - start

    if duration < MIN_DURATION or duration > MAX_DURATION:
        return

    # 🔥 STRICT SINGLE SPEAKER CHECK
    if not is_single_speaker(start, end, speaker, segments):
        rejected_overlap +=1
        return

    s = int(start * sr)
    e = int(end * sr)

    clip = audio[s:e]

    if len(clip) == 0 or not np.any(clip):
        return

    # normalize
    max_val = np.max(np.abs(clip))
    if max_val > 0:
        clip = clip / max_val

    # 🔥 DUPLICATE CHECK (HASH)
    clip_bytes = (clip * 32767).astype("int16").tobytes()
    clip_hash = hashlib.md5(clip_bytes).hexdigest()

    

    if clip_hash in seen_hashes:
        rejected_overlap += 1
        return
    seen_hashes.add(clip_hash)

    # 🔥 CREATE SPEAKER FOLDER
    speaker_dir = os.path.join(OUTPUT_DIR, speaker)
    os.makedirs(speaker_dir, exist_ok=True)

    filename = f"{speaker}_{clip_id:06d}.wav"
    path = os.path.join(speaker_dir, filename)

    sf.write(path, clip, sr)

    clip_id += 1

# splitting logic
for seg in merged:
    start = seg["start"]
    end = seg["end"]
    speaker = seg["speaker"]

    duration = end - start

    # if already good duration
    if TARGET_MIN <= duration <= TARGET_MAX:
        save_clip(start, end, speaker)
        continue

    # split long segments
    curr = start
    while curr < end:
        chunk_end = min(curr + TARGET_MAX, end)
        save_clip(curr, chunk_end, speaker)
        curr = chunk_end

# =========================
# FINAL STATS
# =========================
print(f"\n🎯 Total clips created: {clip_id}")
print(f"❌ Rejected (overlap): {rejected_overlap}")
print(f"❌ Rejected (duplicate): {rejected_duplicate}")
print(f"📁 Output folder: {OUTPUT_DIR}")