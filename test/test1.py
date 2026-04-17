import os
import json
import re
from glob import glob
from tqdm import tqdm
import soundfile as sf

from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset
from transformers import AutoProcessor

# =========================
# CONFIG
# =========================
MODEL_ID = "google/gemma-4-E4B-it"

AUDIO_DIR = "/data/TTS/sherphia/data/raw/Tamil/spring_inx_r2"
TRANSCRIPT_DIR = "/data/TTS/sherphia/transcripted_datas/transcripts_omnilingual_spring_inx_r2"
OUTPUT_DIR = "/data/TTS/sherphia/transcripted_datas/transcripts_corrected_spring_inx_r2"
BATCH_SIZE = 8  # Process N files at once — tune to your VRAM

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD MODEL (ONCE)
# =========================
print("Loading model...")

processor = AutoProcessor.from_pretrained(MODEL_ID)

llm = LLM(
    model=MODEL_ID,
    dtype="bfloat16",
    max_model_len=8192,
    limit_mm_per_prompt={"audio": 1},   # one audio clip per request
    gpu_memory_utilization=0.90,
    trust_remote_code=True,
)

sampling_params = SamplingParams(
    temperature=1.0,
    top_p=0.95,
    top_k=64,
    max_tokens=300,
)

print("Model loaded ✅")

# =========================
# SYSTEM PROMPT
# =========================
SYSTEM_PROMPT = """You are an expert multilingual ASR correction system.

Your task is to convert the given transcript into its fully correct and natural written form.

This is NOT a minimal correction task.

You MUST:
- Rewrite the sentence into its correct grammatical form
- Fix word formations and inflections
- Merge or split words wherever required
- Correct incorrect words using audio context
- Ensure the output is fully natural and fluent
- Fix spacing issues completely

LEXICAL VALIDATION (VERY IMPORTANT):
- You MUST check whether each word is a real, valid word in the language
- If a word is uncommon, incorrect, or does not fit the sentence context, you MUST replace it
- Do NOT preserve words just because they look correct
- If a word is not semantically meaningful in the sentence, correct it

CONTEXT CHECK:
- Every word must make logical and grammatical sense in the sentence
- If a word breaks meaning, replace it with the correct word based on audio + context

STRICT OVERRIDE RULE:
- Even if transcript word seems valid, REPLACE it if:
  1. It does not match audio
  2. It does not fit context
  3. It is not commonly used in written language

Important:
- The output must be in standard written form, not spoken form
- Do not preserve incorrect spacing or structure

Constraints:
- Preserve meaning exactly
- Do not add or remove information

Return ONLY valid JSON:
{
  "corrected": "...",
  "confidence": 0.0,
  "codemix": false,
  "reject": false
}"""

# =========================
# JSON PARSER
# =========================
def safe_json_parse(text):
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            raw = match.group()
            raw = raw.replace("'", '"')
            raw = raw.replace("True", "true").replace("False", "false")
            return json.loads(raw)
    except Exception as e:
        print(f"JSON parse error: {e}\n  Offending text: {text[:300]}")
    return None


# =========================
# BUILD PROMPT FOR ONE FILE
# =========================
def build_prompt(transcript: str, duration: float) -> str:
    user_text = (
        f"Audio duration: {duration:.2f} seconds\n\n"
        f'Raw ASR transcript:\n"{transcript}"\n\n'
        "Instructions:\n"
        "- Listen to the audio carefully\n"
        "- Fully normalize into correct written form\n"
        "- Fix grammar, word forms, spacing\n"
        "- Convert spoken style into written form\n"
        "- Fix joins, splits, grammar, morphology\n"
        "- Return ONLY valid JSON as instructed"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": "<placeholder>"},  # replaced below
                {"type": "text", "text": user_text},
            ],
        },
    ]

    # apply_chat_template renders text; audio token position is handled by vLLM
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


# =========================
# GET FILE LIST
# =========================
audio_files = sorted(glob(os.path.join(AUDIO_DIR, "*.wav")))
print(f"Found {len(audio_files)} audio files")

# Filter to only unprocessed files that have a transcript
pending = []
for audio_path in audio_files:
    file_id = os.path.basename(audio_path).replace(".wav", "")
    output_path = os.path.join(OUTPUT_DIR, f"{file_id}.json")
    if os.path.exists(output_path):
        continue
    transcript_path = os.path.join(TRANSCRIPT_DIR, f"{file_id}.json")
    if not os.path.exists(transcript_path):
        print(f"Missing transcript for {file_id}, skipping...")
        continue
    pending.append((file_id, audio_path, transcript_path))

print(f"{len(pending)} files to process")

# =========================
# BATCH PROCESSING LOOP
# =========================
for batch_start in tqdm(range(0, len(pending), BATCH_SIZE)):
    batch = pending[batch_start : batch_start + BATCH_SIZE]

    prompts = []
    audio_arrays = []
    sample_rates = []
    file_ids = []
    raw_transcripts = []

    for file_id, audio_path, transcript_path in batch:
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            transcript = data.get("transcript", "").strip()
            if not transcript:
                print(f"Empty transcript: {file_id}, skipping...")
                continue

            audio, sr = sf.read(audio_path)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            max_samples = 30 * sr
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            duration = len(audio) / sr

            prompts.append(build_prompt(transcript, duration))
            audio_arrays.append(audio)
            sample_rates.append(sr)
            file_ids.append(file_id)
            raw_transcripts.append(transcript)

        except Exception as e:
            import traceback
            print(f"\n❌ Error loading {file_id}: {e}")
            traceback.print_exc()

    if not prompts:
        continue

    # Build vLLM inputs — each prompt gets its own audio
    vllm_inputs = [
        {
            "prompt": prompt,
            "multi_modal_data": {
                "audio": (audio_arr, sr),  # (numpy_array, sample_rate)
            },
        }
        for prompt, audio_arr, sr in zip(prompts, audio_arrays, sample_rates)
    ]

    try:
        outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)
    except Exception as e:
        import traceback
        print(f"\n❌ vLLM generate error on batch: {e}")
        traceback.print_exc()
        outputs = [None] * len(prompts)

    for i, (file_id, raw_transcript) in enumerate(zip(file_ids, raw_transcripts)):
        output_path = os.path.join(OUTPUT_DIR, f"{file_id}.json")

        result = None
        if outputs[i] is not None:
            response_text = outputs[i].outputs[0].text
            print(f"\n[DEBUG {file_id}] Raw response: {response_text[:200]}")
            result = safe_json_parse(response_text)

        if result is None:
            result = {
                "corrected": raw_transcript,
                "confidence": 0.0,
                "codemix": False,
                "reject": True,
            }

        final_output = {
            "file": f"{file_id}.wav",
            "raw_transcript": raw_transcript,
            "corrected": result.get("corrected", raw_transcript),
            "confidence": result.get("confidence", 0.0),
            "codemix": result.get("codemix", False),
            "reject": result.get("reject", False),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)

print("\n✅ ALL FILES PROCESSED")