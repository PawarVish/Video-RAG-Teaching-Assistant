import json
from pathlib import Path

import joblib
import pandas as pd
import requests


# -----------------------------------
# Project paths
# -----------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

JSON_DIR = PROJECT_ROOT / "data" / "processed" / "json"

EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = EMBEDDINGS_DIR / "embeddings.joblib"


# -----------------------------------
# Ollama configuration
# -----------------------------------

OLLAMA_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "bge-m3"

# Process a small number of texts at a time
BATCH_SIZE = 16


# -----------------------------------
# Create embeddings for one batch
# -----------------------------------

def create_embeddings(text_list):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": EMBEDDING_MODEL,
            "input": text_list
        },
        timeout=300
    )

    if not response.ok:
        print("\nOllama error:")
        print(response.text)

    response.raise_for_status()

    return response.json()["embeddings"]


# -----------------------------------
# Find JSON files
# -----------------------------------

json_files = sorted(JSON_DIR.glob("*.json"))

print(f"Found {len(json_files)} JSON files.")


# -----------------------------------
# Read all chunks
# -----------------------------------

all_chunks = []

for json_file in json_files:

    print(f"Reading: {json_file.name}")

    with open(json_file, "r", encoding="utf-8") as f:
        content = json.load(f)

    for chunk in content["chunks"]:
        all_chunks.append(chunk)


print(f"\nTotal chunks: {len(all_chunks)}")


# -----------------------------------
# Generate embeddings in batches
# -----------------------------------

all_embeddings = []

total_chunks = len(all_chunks)

print("\nCreating BGE-M3 embeddings...")
print(f"Batch size: {BATCH_SIZE}\n")


for start in range(0, total_chunks, BATCH_SIZE):

    end = min(start + BATCH_SIZE, total_chunks)

    batch_chunks = all_chunks[start:end]

    texts = [
        chunk["text"]
        for chunk in batch_chunks
    ]

    print(
        f"Embedding chunks {start + 1}-{end} "
        f"of {total_chunks}..."
    )

    batch_embeddings = create_embeddings(texts)

    all_embeddings.extend(batch_embeddings)


# -----------------------------------
# Attach embeddings
# -----------------------------------

for chunk, embedding in zip(all_chunks, all_embeddings):

    chunk["embedding"] = embedding


# -----------------------------------
# Create DataFrame
# -----------------------------------

df = pd.DataFrame.from_records(all_chunks)


# -----------------------------------
# Save
# -----------------------------------

joblib.dump(df, OUTPUT_FILE)


print("\n" + "=" * 60)
print("EMBEDDINGS CREATED SUCCESSFULLY")
print("=" * 60)

print(f"Total chunks: {len(df)}")
print(f"Embedding dimension: {len(df.iloc[0]['embedding'])}")
print(f"Saved to: {OUTPUT_FILE}")