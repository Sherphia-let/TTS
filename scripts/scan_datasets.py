import os
import json
from datasets import load_dataset, Audio
from tqdm import tqdm

BASE_DIR = "raw/Hindi"
os.makedirs(BASE_DIR, exist_ok=True)


# 🧠 CONFIG — ADD DATASETS HERE
DATASETS = [
    # {
    #      "hf_name": "ai4bharat/Shrutilipi",
    #      "subset": 'hindi',
    #      "folder": "shrutilipi"
    # },
    # {
    #     "hf_name": "utkarsh2299/hindi_indic_TTS",
    #     "subset": None,
    #     "folder": "common_voice_hi"
    # }
        # {
        #     "hf_name": "SPRINGLab/IndicTTS-Hindi",
        #     "subset": None,
        #     "folder": "hindi_tts_datasets"
        # }
    # {
    #     "hf_name": "ai4bharat/indicvoices_r",
    #      "subset": 'Hindi',
    #      "folder": "indicvoices_r_hi"
    #  }
    {
           "hf_name": "SPRINGLab/Hindi-1482Hrs",
            "subset": None,
            "folder": "spring_hindi_1482hrs"
    }
    # {
    # "hf_name": "ai4bharat/indicvoices_r",
    #  "subset": 'Tamil',
    #  "folder": "indicvoices_r"
    # }
    # {
    #     "hf_name": "deepdml/openslr65-tamil",
    #     "subset": None,
    #     "folder": "openslr_65"
    # }
    # {
    #     "hf_name": "ai4bharat/Rasa",
    #  "subset": 'Tamil',
    #  "folder": "rasa_ta"
    # }
    # {
    #     "hf_name": "SPRINGLab/IndicTTS_Tamil",
    #  "subset": None,
    #  "folder": "indic_tts_tamil"
    # }
    # {
    #     "hf_name": "deepdml/iisc-mile-tamil-asr",
    #      "subset": None,
    #      "folder": "iisc_mile"
    # }
    # {
    #     "hf_name": "SPRINGLab/SPRING_INX_Tamil_R2",
    #  "subset": None,
    #  "folder": "spring_inx_r2"
    # }
    # {
    #     "hf_name": "SPRINGLab/SPRING_INX_Tamil_R1",
    #      "subset": None,
    #      "folder": "spring_inx_r1"
    # }
]


MANIFEST_PATH = "manifest.json"

if os.path.exists(MANIFEST_PATH) and os.path.getsize(MANIFEST_PATH) > 0:
    try:
        with open(MANIFEST_PATH, "r") as f:
            manifest = json.load(f)
    except json.JSONDecodeError:
        print("⚠️ Corrupted manifest, starting fresh")
        manifest = []
else:
    manifest = []

# ✅ Track existing paths to avoid duplicates
existing_paths = set(item["path"] for item in manifest)



def disable_decoding(ds):
    for col in ds.features:
        if isinstance(ds.features[col], Audio):
            ds = ds.cast_column(col, Audio(decode=False))
    return ds


def extract_audio_bytes(sample):
    """Universal extractor"""
    if "audio_filepath" in sample and sample["audio_filepath"]:
        return sample["audio_filepath"].get("bytes")

    if "audio" in sample and sample["audio"]:
        return sample["audio"].get("bytes")

    return None


def get_extension(sample):
    """detect file type"""
    if "audio_filepath" in sample and sample["audio_filepath"]:
        path = sample["audio_filepath"].get("path", "")
        return os.path.splitext(path)[-1] or ".flac"

    if "audio" in sample and sample["audio"]:
        path = sample["audio"].get("path", "")
        return os.path.splitext(path)[-1] or ".wav"

    return ".wav"


def process_dataset(config):
    hf_name = config["hf_name"]
    subset = config["subset"]
    folder = config["folder"]
    limit = 100

    save_dir = os.path.join(BASE_DIR, folder)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n🚀 Processing: {hf_name}")

    # load dataset

    ds = load_dataset(hf_name, subset, split="train", streaming=True)

    ds = disable_decoding(ds)

    existing_files = len(os.listdir(save_dir))
    count = existing_files

    for i, sample in enumerate(tqdm(ds)):

        if limit and i >= limit:
            break

        audio_bytes = extract_audio_bytes(sample)

        if audio_bytes is None:
            continue

        ext = get_extension(sample)
        file_path = os.path.join(save_dir, f"{count}{ext}")

        file_path = file_path.replace("\\", "/")


        # ❌ skip duplicates
        if file_path in existing_paths:
           continue

        try:
            with open(file_path, "wb") as f:
                f.write(audio_bytes)

            manifest.append({
                "source": folder,
                "language": "Hindi",
                "path": file_path,
                "type": "dataset"
            })
            existing_paths.add(file_path)

            count += 1

        except Exception as e:
            print(f"⚠️ Error saving file: {e}")

    print(f"✅ {folder}: {count} files saved")


# 🚀 ENTRY POINT
if __name__ == "__main__":
    for dataset in DATASETS:
        process_dataset(dataset)

    with open("manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n✅ FINAL manifest.json created")
