import os
import json
from glob import glob
from tqdm import tqdm
import soundfile as sf

from transformers import AutoProcessor
from vllm import LLM, SamplingParams

# =========================
# CONFIG
# =========================
os.environ["VLLM_USE_DEEP_GEMM"] = "0"

MODEL_ID = "google/gemma-4-E4B-it"

INPUT_DIR   = "/data/TTS/sherphia/transcripted_datas/transcripts_corrected_hindi_movies"
AUDIO_DIR   = "/data/TTS/sherphia/data/final_clips"
WHISPER_DIR = "/data/TTS/sherphia/transcripted_datas/transcripts_whisper_hindi_clean_v3"
OUTPUT_DIR  = "/data/TTS/sherphia/data/clean_v3/hindi/transcripts_normalized_hindi_movies"

MAX_AUDIO_SEC = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD MODEL (ONCE) — vLLM style
# =========================
print("Loading model with vLLM...")

processor = AutoProcessor.from_pretrained(MODEL_ID)

llm = LLM(
    model=MODEL_ID,
    dtype="bfloat16",
    limit_mm_per_prompt={"audio": 1},
)

sampling_params = SamplingParams(
    temperature=1.0,
    top_p=0.95,
    top_k=64,
    max_tokens=256,
)

print("Model loaded ✅")

# =========================
# SYSTEM PROMPT
# =========================
SYSTEM_PROMPT = """You are a multilingual text normalization expert.

Your task:
- Convert ASR transcript into clean, grammatically correct language
- Fix spelling, phonetic errors, spacing, and morphology
- Use audio to resolve ambiguities
- Convert numbers into correct spoken language form.
- Convert any English-origin words written in non-English scripts into standard English spelling, without changing the rest of the sentence.
- Strictly dont write any non english script words into english script.
"""

# =========================
# SINGLE FILE INFERENCE — vLLM style
# =========================
def normalize_one(text, audio, sr, duration):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": f"""Audio duration: {duration:.2f} seconds

ASR TEXT:
{text}

TASK:
- Fix grammar, spelling, spacing
- Use audio for correctness
- Keep wording as close as possible

OUTPUT (corrected sentence only):"""}
            ]
        }
    ]

    # Chat template formatting (processor only, no tensors)
    text_input = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    # vLLM multimodal input
    prompt_input = {
        "prompt": text_input,
        "multi_modal_data": {
            "audio": [(audio, sr)]
        },
    }

    outputs = llm.generate(prompt_input, sampling_params=sampling_params)
    return outputs[0].outputs[0].text.strip()

# =========================
# LOAD WHISPER TRANSCRIPT
# =========================
def load_whisper_transcript(file_name):
    whisper_path = os.path.join(
        WHISPER_DIR,
        os.path.splitext(file_name)[0] + ".json"
    )

    if not os.path.exists(whisper_path):
        whisper_path = os.path.join(
            WHISPER_DIR,
            os.path.basename(os.path.splitext(file_name)[0]) + ".json"
        )
    if not os.path.exists(whisper_path):
        return ""
    try:
        with open(whisper_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("transcript", "")
    except:
        return ""

# =========================
# PROCESS ONE JSON FILE
# =========================
def process_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    results = []

    for item in tqdm(data, desc=os.path.basename(input_path)):
        file_name = item.get("file", "")
        if not file_name:
            continue

        audio_path = os.path.join(AUDIO_DIR, file_name)

        if not os.path.exists(audio_path):
            # fallback → try basename only
            audio_path = os.path.join(AUDIO_DIR, os.path.basename(file_name))

        if not os.path.exists(audio_path):
            print(f"Missing audio: {file_name}, skipping...")
            continue

        try:
            audio, sr = sf.read(audio_path)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            max_samples = MAX_AUDIO_SEC * sr
            if len(audio) > max_samples:
                audio = audio[:max_samples]

            duration = len(audio) / sr

            raw_text = load_whisper_transcript(file_name)

            corrected_input = item.get("corrected", item.get("transcript", "")).strip()
            if not corrected_input:
                print(f"Empty text for {file_name}, skipping...")
                continue

            normalized = normalize_one(corrected_input, audio, sr, duration)

            results.append({
                "file": file_name,
                "raw_text": raw_text,
                "normalized_text": normalized
            })

        except Exception as e:
            import traceback
            print(f"\n❌ Error on {file_name}: {e}")
            traceback.print_exc()
            results.append({
                "file": file_name,
                "raw_text": "",
                "normalized_text": item.get("corrected", ""),
                "error": str(e)
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved → {output_path}")

# =========================
# MAIN
# =========================
def main():
    input_files = []

    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.endswith(".json"):
                input_files.append(os.path.join(root, file))

    input_files = sorted(input_files)
    print(f"Found {len(input_files)} files")

    for input_path in input_files:
        rel_path = os.path.relpath(input_path, INPUT_DIR)
        output_path = os.path.join(OUTPUT_DIR, rel_path)

        filename = os.path.basename(input_path)

        if os.path.exists(output_path):
            print(f"Already done, skipping: {filename}")
            continue

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        print(f"\nProcessing: {filename}")
        process_file(input_path, output_path)

    print("\n✅ All files processed successfully!")

if __name__ == "__main__":
    main()