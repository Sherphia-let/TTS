import os
import numpy as np
import soundfile as sf
import json

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_AUDIO = os.path.join(BASE_DIR, "data/raw/movies/hindi/housefull.wav")
OUTPUT_AUDIO = os.path.join(BASE_DIR, "data/no_songs/movie_clean.wav")
LOG_FILE = os.path.join(BASE_DIR, "data/no_songs/removed_segments.json")

os.makedirs(os.path.dirname(OUTPUT_AUDIO), exist_ok=True)

# 🔥 TUNED PARAMETERS (IMPORTANT)
WINDOW_SIZE = 2.0      # seconds
HOP_SIZE = 1.0         # seconds

ENERGY_THRESHOLD = 0.008   # lowered for movie audio
MIN_SONG_DURATION = 8      # shorter detection
BUFFER = 4                 # remove edges
MERGE_GAP = 5              # merge nearby segments

# =========================
# LOAD AUDIO
# =========================
print("🎧 Loading audio...")

audio, sr = sf.read(INPUT_AUDIO)

# Convert to mono
if audio.ndim == 2:
    audio = np.mean(audio, axis=1)

total_duration = len(audio) / sr
print(f"📏 Duration: {total_duration:.2f} sec")

# =========================
# ANALYZE WINDOWS
# =========================
print("🔍 Detecting song segments...")

window_samples = int(WINDOW_SIZE * sr)
hop_samples = int(HOP_SIZE * sr)

candidates = []
current_start = None

for start in range(0, len(audio) - window_samples, hop_samples):
    segment = audio[start:start + window_samples]

    energy = np.mean(segment ** 2)

    # DEBUG (optional)
    # print(f"Energy: {energy:.5f}")

    if energy > ENERGY_THRESHOLD:
        if current_start is None:
            current_start = start / sr
    else:
        if current_start is not None:
            end_time = start / sr

            if (end_time - current_start) > MIN_SONG_DURATION:
                candidates.append((current_start, end_time))

            current_start = None

# Last segment
if current_start is not None:
    end_time = total_duration
    if (end_time - current_start) > MIN_SONG_DURATION:
        candidates.append((current_start, end_time))

print(f"🟡 Raw detected segments: {len(candidates)}")

# =========================
# MERGE SEGMENTS
# =========================
def merge_segments(segments, gap):
    if not segments:
        return []

    segments = sorted(segments)
    merged = [segments[0]]

    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]

        if start - prev_end < gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    return merged

song_regions = merge_segments(candidates, MERGE_GAP)

# =========================
# APPLY BUFFER
# =========================
buffered_regions = []

for start, end in song_regions:
    start = max(0, start - BUFFER)
    end = min(total_duration, end + BUFFER)
    buffered_regions.append((start, end))

print(f"🎵 Final detected song segments: {len(buffered_regions)}")

# =========================
# REMOVE SONG SEGMENTS
# =========================
print("✂️ Removing song segments...")

clean_audio = []
current_pos = 0
removed_log = []

for start, end in buffered_regions:
    start_sample = int(start * sr)
    end_sample = int(end * sr)

    # Keep non-song part
    clean_audio.append(audio[current_pos:start_sample])

    removed_log.append({
        "start": round(start, 2),
        "end": round(end, 2),
        "duration": round(end - start, 2)
    })

    current_pos = end_sample

# Add remaining audio
clean_audio.append(audio[current_pos:])
clean_audio = np.concatenate(clean_audio)

# =========================
# SAVE OUTPUT
# =========================
print("💾 Saving cleaned audio...")

sf.write(OUTPUT_AUDIO, clean_audio, sr)

with open(LOG_FILE, "w") as f:
    json.dump(removed_log, f, indent=2)

print("\n✅ Song removal complete!")
print(f"🎧 Clean audio: {OUTPUT_AUDIO}")
print(f"📝 Removed segments: {LOG_FILE}")