import json
from pathlib import Path
import whisper


# -----------------------------------
# Project paths
# -----------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

AUDIO_DIR = PROJECT_ROOT / "data" / "processed" / "trimmed_audio"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------
# Load Whisper Base
# -----------------------------------

print("Loading Whisper base model...")

model = whisper.load_model("base")

print("Whisper model loaded.\n")


# -----------------------------------
# Find all MP3 files
# -----------------------------------

audio_files = sorted(AUDIO_DIR.glob("*.mp3"))

print(f"Found {len(audio_files)} audio files.\n")


# -----------------------------------
# Process each audio file
# -----------------------------------

for audio_file in audio_files:

    # Example:
    # 01_Installing VS Code & How Websites Work.mp3

    parts = audio_file.stem.split("_", 1)

    number = parts[0]
    title = parts[1] if len(parts) > 1 else audio_file.stem

    output_file = OUTPUT_DIR / f"{audio_file.stem}.json"

    print("=" * 60)
    print(f"Video: {number}")
    print(f"Title: {title}")
    print(f"Audio: {audio_file.name}")
    print("=" * 60)

    # Skip already processed files
    if output_file.exists():
        print("JSON already exists — skipping.\n")
        continue

    # Hindi/Hinglish → English
    result = model.transcribe(
        str(audio_file),
        language="hi",
        task="translate",
        fp16=False
    )

    # Create timestamped chunks
    chunks = []

    for segment in result["segments"]:

        chunks.append({
            "number": number,
            "title": title,
            "start": round(segment["start"], 2),
            "end": round(segment["end"], 2),
            "text": segment["text"].strip()
        })

    # Final JSON structure
    data = {
        "chunks": chunks,
        "text": result["text"].strip()
    }

    # Save JSON
    with open(output_file, "w", encoding="utf-8") as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"Completed Video {number}")
    print(f"Chunks: {len(chunks)}\n")


print("=" * 60)
print("ALL AVAILABLE AUDIO FILES PROCESSED")
print("=" * 60)