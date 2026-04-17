import os
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_dir = os.path.join(BASE_DIR, "D:\Eros_tasks\text_to_speach\data\separated\test\htdemucs_ft")
output_dir = os.path.join(BASE_DIR, "data/separated/movies_clean")

os.makedirs(output_dir, exist_ok=True)

for root, _, files in os.walk(input_dir):
    for file in files:
        if file == "vocals.wav":
            input_path = os.path.join(root, file)

            # folder name = chunk name
            chunk_name = os.path.basename(root)

            output_path = os.path.join(output_dir, f"{chunk_name}.wav")

            shutil.copy(input_path, output_path)

            print(f"✅ Collected: {chunk_name}.wav")

print("🎯 All vocals collected!")