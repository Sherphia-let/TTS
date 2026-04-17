from datasets import load_dataset, Audio, get_dataset_config_names
import os
import subprocess
import soundfile as sf
import io
from tqdm import tqdm

# =========================
# CONFIG
# =========================
DATASET_NAME = "ai4bharat/indicvoices_r"  
TARGET_LANG = "ta"   # change if needed
SPLIT = "train[:100]"

OUTPUT_DIR = "data/raw/tamil/indicvoices_r/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD DATASET (AUTO-HANDLE CONFIG)
# =========================
def load_any_dataset(name, lang, split):
    try:
        configs = get_dataset_config_names(name)
        configs_lower = [c.lower() for c in configs]

        print(f"📚 Multiple configs found: {configs}")

        # normalize target language
        lang = lang.lower()
        # optional: handle code aliases
        lang_aliases = {
            "ta": ["ta", "tamil","Ta","Tamil"]
        }
        possible = lang_aliases.get(lang, [lang])

        # find matching config
        selected = None
        for idx, c in enumerate(configs_lower):
            if c in possible:
                selected = configs[idx]  # use original case-sensitive name
                break

        if selected:
            print(f"✅ Using language config: {selected}")
            return load_dataset(name, selected, split=split)
        else:
            print(f"⚠️ Language '{lang}' not found. Using default config")
            return load_dataset(name, configs[0], split=split)

    except Exception as e:
        print(f"⚠️ Could not fetch configs: {e}")
        print("➡️ Falling back to normal loading")
        return load_dataset(name, split=split)

dataset = load_any_dataset(DATASET_NAME, TARGET_LANG, SPLIT)

# Ensure audio column behaves consistently
if "audio" in dataset.column_names:
    dataset = dataset.cast_column("audio", Audio(decode=False))
else:
    raise ValueError("❌ No 'audio' column found in dataset")

# =========================
# FFMPEG CONVERSION
# =========================
def convert_to_24k_mono(input_path, output_path):
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", "24000",
        "-sample_fmt", "s16",
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# =========================
# PROCESS LOOP
# =========================
for i, sample in enumerate(tqdm(dataset)):

    audio = sample["audio"]
    text = sample.get("text", "")

    temp_input = f"{OUTPUT_DIR}/temp_{i}.wav"
    output_path = f"{OUTPUT_DIR}/clip_{i}.wav"

    try:
        # -------------------------
        # CASE 1: BYTES
        # -------------------------
        if isinstance(audio, dict) and audio.get("bytes") is not None:
            data, sr = sf.read(io.BytesIO(audio["bytes"]))
            sf.write(temp_input, data, sr)

        # -------------------------
        # CASE 2: FILE PATH
        # -------------------------
        elif isinstance(audio, dict) and audio.get("path") is not None:
            temp_input = audio["path"]

        # -------------------------
        # CASE 3: Already decoded (rare)
        # -------------------------
        elif isinstance(audio, dict) and "array" in audio:
            sf.write(temp_input, audio["array"], audio["sampling_rate"])

        else:
            print(f"❌ Unsupported audio format at {i}")
            continue

    except Exception as e:
        print(f"❌ Failed decoding {i}: {e}")
        continue

    # -------------------------
    # CONVERT AUDIO
    # -------------------------
    convert_to_24k_mono(temp_input, output_path)

    # -------------------------
    # CLEANUP (only if temp file we created)
    # -------------------------
    if os.path.exists(temp_input) and "temp_" in temp_input:
        os.remove(temp_input)

    # -------------------------
    # VERIFY
    # -------------------------
    if not os.path.exists(output_path):
        print(f"❌ FFmpeg failed {i}")
    else:
        print(f"✅ Created {output_path}")

print("\n✅ DONE — universal dataset pipeline working")