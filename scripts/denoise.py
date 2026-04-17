import os
import subprocess

# =========================
# PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_dir = os.path.join(BASE_DIR, "data/final_clips")
output_dir = os.path.join(BASE_DIR, "data/denoised_clips")

os.makedirs(output_dir, exist_ok=True)

# =========================
# ENV SETUP
# =========================
def setup_environment():
    env_python = os.path.join(BASE_DIR, "denoise_env", "Scripts", "python.exe")

    if not os.path.exists(env_python):
        raise FileNotFoundError(f"❌ Env python not found: {env_python}")

    print(f"🐍 Using Python: {env_python}")
    return env_python


# =========================
# DENOISE FUNCTION
# =========================
def denoise_with_df(env_python, input_path, output_path):

    wrapper_script = f'''
import torch
import torchaudio
import soundfile as sf
import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    torchaudio.set_audio_backend("soundfile")
except:
    pass

def run(input_path, output_path):
    try:
        import df

        audio, sr = sf.read(input_path)

        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        else:
            audio = audio.T

        audio = torch.from_numpy(audio).float()

        target_sr = 48000
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            audio = resampler(audio)

        model, df_state, _ = df.init_df(log_level='ERROR')

        with torch.no_grad():
            enhanced = df.enhance(model, df_state, audio, pad=True)

        enhanced = enhanced.cpu().numpy()

        if enhanced.shape[0] > 1:
            enhanced = enhanced.mean(axis=0)
        else:
            enhanced = enhanced[0]

        enhanced = np.clip(enhanced, -1.0, 1.0)

        sf.write(output_path, enhanced, target_sr)
        print("Success")

    except Exception as e:
        print(f"Error: {{e}}")
        exit(1)

if __name__ == "__main__":
    import sys
    run(sys.argv[1], sys.argv[2])
'''

    wrapper_path = os.path.join(BASE_DIR, "temp_denoise.py")

    with open(wrapper_path, "w") as f:
        f.write(wrapper_script)

    try:
        result = subprocess.run(
            [env_python, wrapper_path, input_path, output_path],
            capture_output=True,
            text=True,
            timeout=120
        )

        return result.returncode == 0

    finally:
        if os.path.exists(wrapper_path):
            os.remove(wrapper_path)


# =========================
# MAIN
# =========================
def main():
    env_python = setup_environment()

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

            print(f"🎧 Denoising: {file}")

            if os.path.exists(output_path):
                os.remove(output_path)

            if denoise_with_df(env_python, input_path, output_path):
                processed += 1
            else:
                failed += 1

    print("\n🎯 Denoising complete!")
    print(f"✅ Processed: {processed}")
    print(f"❌ Failed: {failed}")


if __name__ == "__main__":
    main()