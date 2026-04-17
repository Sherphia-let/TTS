import os
import csv
import json
from collections import defaultdict

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_RTTM = os.path.join(BASE_DIR, "data", "annotations", "vocals.rttm")
CSV_OUT = os.path.join(BASE_DIR, "data", "annotations", "speaker_durations.csv")
JSON_OUT = os.path.join(BASE_DIR, "data", "annotations", "speaker_durations.json")
VALID_OUT = os.path.join(BASE_DIR, "data", "annotations", "valid_speakers.json")

# =========================
# CONFIG
# =========================
MIN_DURATION = 180  # seconds (3 minutes)

# manually remove bad speakers (silence / noise)
EXCLUDE_SPEAKERS = {
    "SPEAKER_04"
}

# =========================
# PARSE RTTM
# =========================
def parse_rttm_line(line):
    parts = line.strip().split()
    if len(parts) < 8 or parts[0] != "SPEAKER":
        return None
    try:
        duration = float(parts[4])
        speaker_id = parts[7]
    except (ValueError, IndexError):
        return None
    return speaker_id, duration

# =========================
# MAIN
# =========================
def main():
    totals = defaultdict(float)

    # ---- Load RTTM ----
    with open(INPUT_RTTM, "r", encoding="utf-8") as rttm_file:
        for line in rttm_file:
            parsed = parse_rttm_line(line)
            if parsed is None:
                continue
            speaker_id, duration = parsed
            totals[speaker_id] += duration

    # ---- Build rows ----
    rows = []
    for speaker_id, total_seconds in totals.items():
        total_seconds = round(total_seconds, 2)
        rows.append({
            "speaker_id": speaker_id,
            "total_seconds": total_seconds,
            "total_minutes": round(total_seconds / 60.0, 2),
        })

    # Sort by duration (descending)
    rows.sort(key=lambda item: item["total_seconds"], reverse=True)

    # =========================
    # SAVE ALL SPEAKERS
    # =========================
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["speaker_id", "total_seconds", "total_minutes"],
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(JSON_OUT, "w", encoding="utf-8") as json_file:
        json.dump(rows, json_file, indent=2)

    print("📊 Speaker durations (descending):")
    for row in rows:
        print(f"{row['speaker_id']}: {row['total_seconds']:.2f} sec ({row['total_minutes']:.2f} min)")

    print(f"\n📄 Saved CSV: {CSV_OUT}")
    print(f"📄 Saved JSON: {JSON_OUT}")

    # =========================
    # FILTER VALID SPEAKERS
    # =========================
    valid_speakers = [
        row["speaker_id"]
        for row in rows
        if row["total_seconds"] >= MIN_DURATION
        and row["speaker_id"] not in EXCLUDE_SPEAKERS
    ]

    print("\n✅ Final Valid Speakers (≥ 3 min & cleaned):")
    for spk in valid_speakers:
        print(spk)

    # Save valid speakers
    with open(VALID_OUT, "w", encoding="utf-8") as f:
        json.dump(valid_speakers, f, indent=2)

    print(f"\n📄 Saved valid speakers: {VALID_OUT}")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    main()