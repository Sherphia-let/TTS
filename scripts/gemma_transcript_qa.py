import os
import json
import re
from glob import glob
from tqdm import tqdm
import soundfile as sf
import torch

from transformers import AutoProcessor, AutoModelForCausalLM

# =========================
# CONFIG
# =========================
MODEL_ID = "google/gemma-4-E4B-it"

AUDIO_DIR = "/data/TTS/sherphia/data/raw/Tamil/spring_inx_r2"
TRANSCRIPT_DIR = "/data/TTS/sherphia/transcripted_datas/transcripts_omnilingual_spring_inx_r2"
OUTPUT_DIR = "/data/TTS/sherphia/transcripted_datas/transcripts_corrected_spring_inx_r2"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD MODEL (ONCE)
# =========================
print("Loading model...")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

print("Model loaded ✅")

# =========================
# SYSTEM PROMPT
# =========================
SYSTEM_PROMPT = """
You are an expert multilingual ASR correction system.

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
}
"""

# =========================
# JSON PARSER
# =========================
def safe_json_parse(text):
    if not text:
        return None
    try:
        # Try direct parse first
        return json.loads(text)
    except:
        pass
    try:
        # Extract JSON block
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            raw = match.group()
            # Fix single quotes → double quotes
            raw = raw.replace("'", '"')
            # Fix Python booleans → JSON booleans
            raw = raw.replace("True", "true").replace("False", "false")
            return json.loads(raw)
    except Exception as e:
        print(f"JSON parse error: {e}")
        print(f"  Offending text: {text[:300]}")
    return None

# =========================
# GET FILE LIST (ORDERED)
# =========================
audio_files = sorted(glob(os.path.join(AUDIO_DIR, "*.wav")))
print(f"Found {len(audio_files)} audio files")

# =========================
# MAIN LOOP
# =========================
for audio_path in tqdm(audio_files):

    file_id = os.path.basename(audio_path).replace(".wav", "")
    output_path = os.path.join(OUTPUT_DIR, f"{file_id}.json")

    # Skip already processed
    if os.path.exists(output_path):
        continue

    transcript_path = os.path.join(TRANSCRIPT_DIR, f"{file_id}.json")
    if not os.path.exists(transcript_path):
        print(f"Missing transcript for {file_id}, skipping...")
        continue

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

        USER_PROMPT = f"""Audio duration: {duration:.2f} seconds

Raw ASR transcript:
"{transcript}"

Instructions:
- Listen to the audio carefully
- Fully normalize into correct written form
- Fix grammar, word forms, spacing
- Convert spoken style into written form
- Fix joins, splits, grammar, morphology
- Return ONLY valid JSON as instructed"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": USER_PROMPT}
                ]
            }
        ]

        text_input = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = processor(
            text=text_input,
            audio=audio,
            sampling_rate=sr,
            return_tensors="pt"
        ).to(model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
                do_sample=True
            )

        response = processor.decode(
            output_ids[0][input_len:],
            skip_special_tokens=True
        )

        print(f"\n[DEBUG {file_id}] Raw response: {response[:200]}")

        result = safe_json_parse(response)

    except Exception as e:
        import traceback
        print(f"\n❌ Error on {file_id}: {e}")
        traceback.print_exc()
        result = None

    if result is None:
        result = {
            "corrected": transcript if 'transcript' in dir() else "",
            "confidence": 0.0,
            "codemix": False,
            "reject": True
        }

    final_output = {
        "file": f"{file_id}.wav",
        "raw_transcript": transcript,
        "corrected": result.get("corrected", transcript),
        "confidence": result.get("confidence", 0.0),
        "codemix": result.get("codemix", False),
        "reject": result.get("reject", False)
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

print("\n✅ ALL FILES PROCESSED")