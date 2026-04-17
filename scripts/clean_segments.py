import json
import os
#  these file is created to update the segments.jsonl file to only include the segments from the full movie (vocals) and exclude any segments from the separated vocals. This is necessary because the separated vocals may have different timings and may not align with the original segments, which could cause issues in downstream processing. By keeping only the segments from the full movie, we ensure that our annotations are consistent and accurate for the original audio.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT = os.path.join(BASE_DIR, "data/annotations/segments.jsonl")
OUTPUT = os.path.join(BASE_DIR, "data/annotations/segments_clean.jsonl")

with open(INPUT, "r") as f:
    lines = f.readlines()

clean = []

for line in lines:
    data = json.loads(line)
    
    if data["file"] == "vocals":   # ✅ KEEP ONLY FULL MOVIE
        clean.append(data)

with open(OUTPUT, "w") as f:
    for item in clean:
        f.write(json.dumps(item) + "\n")

print(f"✅ Cleaned segments saved: {OUTPUT}")
print(f"Total kept: {len(clean)}")