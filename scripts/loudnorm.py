import os
import subprocess
import json

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_dir = os.path.join(BASE_DIR, "data/denoised_clips")
output_dir = os.path.join(BASE_DIR, "data/normed_clips")

os.makedirs(output_dir, exist_ok=True)

# =========================
# MAIN
# =========================
processed = 0
failed = 0

for speaker in os.listdir(input_dir):
    speaker_path = os.path.join(input_dir, speaker)

    if not os.path.isdir(speaker_path):
        continue

    out_speaker_path = os.path.join(output_dir, speaker)
    os.makedirs(out_speaker_path, exist_ok=True)

    print(f"\n🎤 Processing Speaker: {speaker}")

    for file in os.listdir(speaker_path):
        if not file.endswith(".wav"):
            continue

        input_path = os.path.join(speaker_path, file)
        output_path = os.path.join(out_speaker_path, file)

        print(f"🔊 Normalizing: {file}")

        try:
            # =========================
            # PASS 1: ANALYSIS
            # =========================
            command1 = [
                "ffmpeg",
                "-i", input_path,
                "-af", "loudnorm=I=-23:TP=-3.0:LRA=11:print_format=json",
                "-f", "null",
                "-"
            ]

            result = subprocess.run(command1, capture_output=True, text=True)

            stderr = result.stderr
            start = stderr.find("{")
            end = stderr.rfind("}") + 1

            if start == -1 or end == -1:
                raise ValueError("Failed to extract loudnorm stats")

            stats = json.loads(stderr[start:end])

            # =========================
            # PASS 2: APPLY
            # =========================
            command2 = [
                "ffmpeg",
                "-y",  # overwrite
                "-i", input_path,
                "-af",
                (
                    f"loudnorm=I=-16:TP=-1.5:LRA=11:"
                    f"measured_I={stats['input_i']}:"
                    f"measured_LRA={stats['input_lra']}:"
                    f"measured_TP={stats['input_tp']}:"
                    f"measured_thresh={stats['input_thresh']}:"
                    f"offset={stats['target_offset']}:"
                    f"linear=true:print_format=summary"
                ),
                output_path
            ]

            subprocess.run(command2, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            processed += 1

        except Exception as e:
            print(f"❌ Failed: {file} | {e}")
            failed += 1

print("\n🎯 Loudness normalization complete!")
print(f"✅ Processed: {processed}")
print(f"❌ Failed: {failed}")