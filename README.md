# Video RAG Teaching Assistant

A resource-efficient Retrieval-Augmented Generation (RAG) system that allows users to ask questions about course videos and receive answers grounded in timestamped video transcripts.

The system converts video/audio content into transcripts, creates semantic embeddings, retrieves relevant transcript chunks, and uses a local Large Language Model to generate grounded responses.

## Features

- Video-to-audio conversion using FFmpeg
- Speech-to-text transcription using OpenAI Whisper
- Timestamp-preserving transcripts
- Transcript chunking
- Semantic retrieval using BGE-M3 embeddings
- Cosine similarity-based retrieval
- Grounding validation for generated answers
- Out-of-course question detection
- Source video and timestamp display
- Local LLM-based answer generation using Llama 3.2 1B
- Streamlit-based interactive interface
- Fully local inference pipeline

## Architecture

![System Architecture](docs/architecture.png)

### End-to-End Pipeline

```text
VIDEO / AUDIO
     |
     v
FFmpeg
     |
     v
MP3 Audio
     |
     v
Whisper
     |
     v
Timestamped JSON
     |
     v
Transcript Chunking
     |
     v
BGE-M3 Embeddings
     |
     v
Knowledge Base
     |
     |
USER QUESTION
     |
     v
BGE-M3 Query Embedding
     |
     v
Cosine Similarity
     |
     v
Relevant Chunks
     |
     v
Llama 3.2 1B
     |
     v
Answer Generation
     |
     v
Grounding Validation
     |
     v
Answer + Source + Timestamp
```

## How It Works

### 1. Video / Audio Processing

Course video or audio content is converted into MP3 format using FFmpeg.

### 2. Speech-to-Text

Whisper processes the audio and generates timestamped transcript segments.

Each transcript segment contains information such as:

- Video number
- Video title
- Start timestamp
- End timestamp
- Transcript text

### 3. Transcript Chunking

The timestamped transcript is divided into smaller chunks so that relevant portions can be retrieved efficiently.

### 4. Embedding Generation

BGE-M3 generates semantic embeddings for the transcript chunks.

The embeddings are stored locally for retrieval.

### 5. Semantic Retrieval

When a user asks a question:

1. The question is converted into a BGE-M3 embedding.
2. Cosine similarity is calculated against the stored transcript embeddings.
3. The most relevant transcript chunks are retrieved.
4. Neighboring chunks are included to preserve additional context.

### 6. Answer Generation

The retrieved transcript evidence is provided to a locally running Llama 3.2 1B model.

The model is instructed to answer using only the retrieved course evidence.

### 7. Grounding Validation

The generated response is checked against the retrieved transcript evidence.

If the generated answer does not meet the grounding requirement, the system falls back to retrieved course evidence instead of presenting an unsupported response.

### 8. Out-of-Course Detection

Questions with insufficient similarity to the course knowledge base are treated as not covered by the available course content.

This prevents unrelated questions from producing irrelevant course sources.

## Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Audio Processing | FFmpeg |
| Speech-to-Text | OpenAI Whisper |
| Embeddings | BGE-M3 |
| Local LLM | Llama 3.2 1B |
| Local Model Runtime | Ollama |
| Retrieval | Cosine Similarity |
| ML Utilities | scikit-learn |
| Data Processing | NumPy, Pandas |
| Storage | Joblib |
| Interface | Streamlit |
| Version Control | Git, GitHub |

## Project Structure

```text
Video-RAG-Teaching-Assistant/
|
├── data/
│   ├── embeddings/
│   ├── processed/
│   │   ├── json/
│   │   └── trimmed_audio/
│   └── raw/
│       ├── audio/
│       └── videos/
|
├── docs/
│   └── architecture.png
|
├── notebooks/
├── outputs/
├── scripts/
│   ├── __init__.py
│   ├── create_embeddings.py
│   ├── rag_assistant.py
│   ├── transcribe_audio.py
│   └── video_to_audio.py
|
├── src/
├── tests/
|
├── .gitignore
├── app.py
├── README.md
└── requirements.txt
```

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/PawarVish/Video-RAG-Teaching-Assistant.git
cd Video-RAG-Teaching-Assistant
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```powershell
pip install -r requirements.txt
```

### 4. Install required external tools

The project requires:

- FFmpeg
- Ollama

### 5. Pull the required Ollama models

```powershell
ollama pull bge-m3
ollama pull llama3.2:1b
```

## Running the Application

Make sure Ollama is running and the required models are available.

Start the Streamlit application:

```powershell
streamlit run app.py
```

The application will open in the browser.

## Example Questions

Examples of questions that can be asked about the available course content include:

```text
What is VS Code?
```

```text
What are HTML headings?
```

```text
What are IDs and classes in HTML?
```

Questions unrelated to the available course content are identified as not covered when their similarity is below the configured retrieval threshold.

## Current Knowledge Base

The current demonstration knowledge base contains:

- 10 processed course audio samples
- Timestamped Whisper transcripts
- 1,175 transcript chunks
- BGE-M3 embeddings
- Local retrieval and generation using Ollama

The project is designed as a resource-efficient local demonstration and does not require cloud APIs for the implemented RAG pipeline.

## Data and Repository Notes

Course media files, generated transcripts, embeddings, and other generated data are excluded from version control.

The repository contains the application code, documentation, configuration, and project structure required to reproduce the implemented pipeline with appropriate source data.

## Design Goals

The project focuses on:

- Local execution
- Resource-efficient processing
- Semantic retrieval
- Grounded answer generation
- Timestamp traceability
- Out-of-course question handling
- Simple and explainable RAG architecture

## Limitations

- The demonstration knowledge base uses a limited set of processed course content.
- Whisper transcription quality can vary depending on audio quality and language.
- Local inference performance depends on available CPU and RAM resources.
- The system is designed for the available course knowledge base and does not provide general-purpose knowledge.

## Future Improvements

Potential future improvements include:

- Processing a larger course collection
- Improving retrieval and reranking
- Adding a persistent vector database
- Supporting additional document and video formats
- Improving answer evaluation
- Adding authentication and multi-user support
- Deploying the application to a cloud environment
