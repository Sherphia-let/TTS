import os
import json
from glob import glob

# مسیر to your folder
INPUT_DIR = "/data/TTS/sherphia/data/transcripts_normalized_iisc_mile"
OUTPUT_FILE = "/data/TTS/sherphia/data/combined_outputs_1.json"

def combine_json_files(input_dir, output_file):
    all_data = []

    # get all json files sorted (0.json → 99.json)
    json_files = sorted(
        glob(os.path.join(input_dir, "*.json")),
        key=lambda x: int(os.path.basename(x).split(".")[0])
    )

    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_data.append(data)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # save combined file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Combined {len(all_data)} files into {output_file}")


if __name__ == "__main__":
    combine_json_files(INPUT_DIR, OUTPUT_FILE)