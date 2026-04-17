import os
import numpy as np
import soundfile as sf
import pandas as pd
import torch

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_dir = os.path.join(BASE_DIR, "data/normed_clips")
clean_dir = os.path.join(BASE_DIR, "data/clean_clips")
reject_dir = os.path.join(BASE_DIR, "data/rejected_clips")
log_file = os.path.join(BASE_DIR, "data/rejected_log.csv")

os.makedirs(clean_dir, exist_ok=True)
os.makedirs(reject_dir, exist_ok=True)

# =========================
# CONFIG
# =========================
MIN_DUR = 2.0
MAX_DUR = 8.0
SNR_TH = 20
CLIP_TH = 0.99
VAD_TH = 0.77

# =========================
# LOAD VAD
# =========================
print("🔊 Loading Silero VAD...")
vad_model, utils = torch.hub.load(
    'snakers4/silero-vad', 'silero_vad', trust_repo=True
)
(get_speech_timestamps, _, _, _, _) = utils

# =========================
# HELPERS
# =========================
def compute_snr(x):
    x = x.astype(np.float32)
    energy = x ** 2

    threshold = np.percentile(energy, 20)
    noise = energy[energy <= threshold]
    signal = energy[energy > threshold]

    if len(noise) == 0 or len(signal) == 0:
        return 0

    noise_power = np.mean(noise)
    signal_power = np.mean(signal)

    if noise_power == 0:
        return 100

    return 10 * np.log10(signal_power / noise_power)


def is_clipped(x):
    return np.max(np.abs(x)) >= CLIP_TH


def speech_ratio(x, sr):
    wav = torch.tensor(x).float()
    timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=sr)

    speech = sum([t['end'] - t['start'] for t in timestamps])
    total = len(x)

    return speech / total if total > 0 else 0


# =========================
# MAIN
# =========================
valid_count = 0
reject_count = 0
rejected_logs = []

for speaker in os.listdir(input_dir):
    speaker_path = os.path.join(input_dir, speaker)

    if not os.path.isdir(speaker_path):
        continue

    out_clean_spk = os.path.join(clean_dir, speaker)
    out_reject_spk = os.path.join(reject_dir, speaker)

    os.makedirs(out_clean_spk, exist_ok=True)
    os.makedirs(out_reject_spk, exist_ok=True)

    print(f"\n🎤 Processing Speaker: {speaker}")

    for file in os.listdir(speaker_path):
        if not file.endswith(".wav"):
            continue

        path = os.path.join(speaker_path, file)

        try:
            audio, sr = sf.read(path)

            if audio.ndim == 2:
                audio = np.mean(audio, axis=1)

            duration = len(audio) / sr
            reason = None

            # (b) duration
            if duration < MIN_DUR or duration > MAX_DUR:
                reason = "bad_duration"

            # empty
            elif len(audio) == 0:
                reason = "empty"

            # (c) clipping
            elif is_clipped(audio):
                reason = "clipping"

            # (a) SNR
            else:
                snr = compute_snr(audio)
                if snr < SNR_TH:
                    reason = "low_snr"

            # (d) VAD
            if reason is None:
                try:
                    ratio = speech_ratio(audio, sr)
                    if ratio < VAD_TH:
                        reason = "low_speech_ratio"
                except:
                    reason = "vad_error"

            # =========================
            # SAVE
            # =========================
            if reason:
                reject_count += 1
                rejected_logs.append([speaker, file, reason])

                out_path = os.path.join(out_reject_spk, file)
                sf.write(out_path, audio, sr)

            else:
                valid_count += 1

                out_path = os.path.join(out_clean_spk, file)
                sf.write(out_path, audio, sr)

        except Exception as e:
            reject_count += 1
            rejected_logs.append([speaker, file, "error"])

# =========================
# SAVE LOG
# =========================
df = pd.DataFrame(rejected_logs, columns=["speaker", "file", "reason"])
df.to_csv(log_file, index=False)

print("\n🎯 Quality filtering complete!")
print(f"✅ Clean clips: {valid_count}")
print(f"❌ Rejected clips: {reject_count}")
print(f"📄 Log saved: {log_file}")