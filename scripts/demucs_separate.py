import os
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_path = os.path.join(
    BASE_DIR,
    "data/no_songs/movie_clean.wav"
)

output_dir = os.path.join(BASE_DIR, "data/separated/movies")

# 🔍 Check if file exists
if not os.path.exists(input_path):
    print("❌ File not found:", input_path)
    exit()

print(f"🎵 Processing: {input_path}")

command = [
    "python", "-m", "demucs",
    "-n", "htdemucs_ft",
    "--two-stems=vocals",
    "-o", output_dir,
    input_path
]

subprocess.run(command, check=True)

print("✅ Demucs done successfully!")