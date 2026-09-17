import re
import joblib
import numpy as np
import requests

from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMBEDDINGS_FILE = (
    PROJECT_ROOT
    / "data"
    / "embeddings"
    / "embeddings.joblib"
)


# ============================================================
# OLLAMA CONFIGURATION
# ============================================================

OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"

EMBEDDING_MODEL = "bge-m3"
LLM_MODEL = "llama3.2:1b"


# ============================================================
# RETRIEVAL CONFIGURATION
# ============================================================

TOP_K = 5
NEIGHBOR_COUNT = 2

SIMILARITY_THRESHOLD = 0.65
GROUNDING_THRESHOLD = 0.40


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

def load_embeddings():
    return joblib.load(EMBEDDINGS_FILE)


# ============================================================
# CREATE QUERY EMBEDDING
# ============================================================

def create_embedding(text):

    response = requests.post(
        OLLAMA_EMBED_URL,
        json={
            "model": EMBEDDING_MODEL,
            "input": [text]
        },
        timeout=300
    )

    response.raise_for_status()

    return response.json()["embeddings"][0]


# ============================================================
# RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve_chunks(query, df):

    question_embedding = create_embedding(query)

    stored_embeddings = np.vstack(
        df["embedding"].values
    )

    similarities = cosine_similarity(
        stored_embeddings,
        [question_embedding]
    ).flatten()

    top_indices = (
        similarities.argsort()[::-1][:TOP_K]
    )

    context_indices = set()

    for index in top_indices:

        start = max(
            0,
            index - NEIGHBOR_COUNT
        )

        end = min(
            len(df),
            index + NEIGHBOR_COUNT + 1
        )

        for i in range(start, end):
            context_indices.add(i)

    context_indices = sorted(
        context_indices
    )

    results = df.iloc[
        context_indices
    ].copy()

    results["similarity"] = similarities[
        context_indices
    ]

    return results


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(results):

    results = results.sort_values(
        "similarity",
        ascending=False
    )

    context_parts = []

    for i, (_, row) in enumerate(
        results.iterrows(),
        start=1
    ):

        context_parts.append(
            f"""
EVIDENCE {i}

Video Number: {row['number']}
Video Title: {row['title']}
Timestamp: {row['start']:.2f}s - {row['end']:.2f}s

Transcript:
{row['text']}
"""
        )

    return "\n".join(
        context_parts
    )


# ============================================================
# GENERATE GROUNDED ANSWER
# ============================================================

def generate_answer(query, context):

    prompt = f"""
You are a teaching assistant for a video course.

Answer the user's question using ONLY the course transcript
evidence provided below.

The transcript is the source of truth.

Use information explicitly supported by the transcript.

You may combine information from multiple evidence sections
when they are relevant to the same question.

Do not use outside knowledge.

Do not invent facts.

Do not invent steps.

Do not add information that is not supported by the transcript.

If the transcript does not provide enough information,
respond exactly with:

The provided course content does not contain enough information to answer this question.

Keep the answer concise and directly answer the question.

COURSE TRANSCRIPT:
{context}

USER QUESTION:
{query}

ANSWER:
"""

    response = requests.post(
        OLLAMA_GENERATE_URL,
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 80
            }
        },
        timeout=300
    )

    response.raise_for_status()

    return response.json()["response"].strip()


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# SPLIT INTO SENTENCES
# ============================================================

def split_sentences(text):

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip()
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ============================================================
# CHECK LLM REFUSAL
# ============================================================

def is_refusal(answer):

    refusal_phrases = [
        "does not contain enough information",
        "not enough information",
        "cannot answer",
        "can't answer",
        "unable to answer",
        "not provided in the course",
        "not available in the transcript",
        "does not provide enough information"
    ]

    answer_lower = answer.lower()

    return any(
        phrase in answer_lower
        for phrase in refusal_phrases
    )


# ============================================================
# GROUNDING SCORE
# ============================================================

def sentence_grounding_score(
    sentence,
    transcript_text
):

    sentence_clean = clean_text(
        sentence
    )

    transcript_clean = clean_text(
        transcript_text
    )

    if not sentence_clean:
        return 0.0

    if not transcript_clean:
        return 0.0

    try:

        vectorizer = TfidfVectorizer()

        vectors = vectorizer.fit_transform(
            [
                sentence_clean,
                transcript_clean
            ]
        )

        tfidf_score = cosine_similarity(
            vectors[0:1],
            vectors[1:2]
        )[0][0]

    except ValueError:

        tfidf_score = 0.0

    stop_words = {
        "the", "a", "an", "is", "are",
        "was", "were", "this", "that",
        "it", "to", "of", "in", "on",
        "for", "and", "or", "you",
        "your", "we", "they", "i",
        "with", "as", "from", "be",
        "can"
    }

    sentence_words = {
        word
        for word in sentence_clean.split()
        if word not in stop_words
    }

    transcript_words = {
        word
        for word in transcript_clean.split()
        if word not in stop_words
    }

    if sentence_words:

        overlap = (
            len(
                sentence_words
                & transcript_words
            )
            / len(sentence_words)
        )

    else:

        overlap = 0.0

    return (
        0.6 * tfidf_score
        + 0.4 * overlap
    )


# ============================================================
# VALIDATE AGAINST COMBINED RETRIEVED EVIDENCE
# ============================================================

def validate_answer(
    answer,
    results
):

    if is_refusal(answer):

        return False, 0.0

    sentences = split_sentences(
        answer
    )

    if not sentences:

        return False, 0.0

    # --------------------------------------------------------
    # Combine all retrieved transcript evidence
    # --------------------------------------------------------

    combined_transcript = " ".join(
        str(text)
        for text in results["text"]
    )

    sentence_scores = []

    for sentence in sentences:

        score = sentence_grounding_score(
            sentence,
            combined_transcript
        )

        sentence_scores.append(
            score
        )

    minimum_score = min(
        sentence_scores
    )

    return (
        minimum_score >= GROUNDING_THRESHOLD,
        minimum_score
    )


# ============================================================
# FIND BEST SOURCE
# ============================================================

def find_answer_source(
    answer,
    results
):

    sentences = split_sentences(
        answer
    )

    if not sentences:

        return results.loc[
            results["similarity"].idxmax()
        ]

    best_index = None
    best_score = -1.0

    for index, row in results.iterrows():

        transcript = str(
            row["text"]
        )

        scores = []

        for sentence in sentences:

            scores.append(
                sentence_grounding_score(
                    sentence,
                    transcript
                )
            )

        average_score = (
            sum(scores)
            / len(scores)
        )

        if average_score > best_score:

            best_score = average_score
            best_index = index

    return results.loc[
        best_index
    ]


# ============================================================
# FALLBACK COURSE EVIDENCE
# ============================================================

def build_fallback_evidence(
    query,
    results
):

    ranked = results.sort_values(
        "similarity",
        ascending=False
    )

    top_results = ranked.head(
        TOP_K
    )

    evidence_parts = []

    for _, row in top_results.iterrows():

        text = str(
            row["text"]
        ).strip()

        if not text:
            continue

        evidence_parts.append(
            text
        )

    if not evidence_parts:

        return (
            "The provided course content does not "
            "contain enough information to answer "
            "this question."
        )

    return (
        "The retrieved course transcript contains "
        "the following relevant evidence:\n\n"
        + "\n\n".join(evidence_parts)
    )


# ============================================================
# MAIN RAG FUNCTION
# ============================================================

def answer_question(
    query,
    df=None
):

    if df is None:

        df = load_embeddings()

    # --------------------------------------------------------
    # RETRIEVE
    # --------------------------------------------------------

    results = retrieve_chunks(
        query,
        df
    )

    # --------------------------------------------------------
    # BEST SEMANTIC RESULT
    # --------------------------------------------------------

    best_semantic_index = (
        results["similarity"].idxmax()
    )

    best_semantic_result = (
        results.loc[
            best_semantic_index
        ]
    )

    best_similarity = float(
        best_semantic_result[
            "similarity"
        ]
    )

    # --------------------------------------------------------
    # OUT-OF-SCOPE CHECK
    # --------------------------------------------------------

    if (
        best_similarity
        < SIMILARITY_THRESHOLD
    ):

        return {
            "answer": (
                "The provided course content does not "
                "contain enough information to answer "
                "this question."
            ),
            "source": None,
            "similarity": best_similarity,
            "grounding_score": 0.0,
            "answer_method": "Not covered",
            "retrieved": results
        }

    # --------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------

    context = build_context(
        results
    )

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    generated_answer = generate_answer(
        query,
        context
    )

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    is_grounded, grounding_score = (
        validate_answer(
            generated_answer,
            results
        )
    )

    # --------------------------------------------------------
    # VALID ANSWER
    # --------------------------------------------------------

    if is_grounded:

        source = find_answer_source(
            generated_answer,
            results
        )

        return {
            "answer": generated_answer,
            "source": source,
            "similarity": best_similarity,
            "grounding_score": grounding_score,
            "answer_method": (
                "Llama 3.2 1B "
                "(grounding validated)"
            ),
            "retrieved": results
        }

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    fallback_answer = build_fallback_evidence(
        query,
        results
    )

    return {
        "answer": fallback_answer,
        "source": best_semantic_result,
        "similarity": best_similarity,
        "grounding_score": grounding_score,
        "answer_method": (
            "Retrieved course evidence "
            "(LLM grounding failed)"
        ),
        "retrieved": results
    }


# ============================================================
# TERMINAL MODE
# ============================================================

def main():

    print("=" * 70)
    print("VIDEO RAG TEACHING ASSISTANT")
    print("=" * 70)

    print("\nLoading embeddings...")

    df = load_embeddings()

    print(
        f"Loaded {len(df)} chunks."
    )

    print("\nModels:")

    print(
        f"Embedding model: "
        f"{EMBEDDING_MODEL}"
    )

    print(
        f"LLM model: "
        f"{LLM_MODEL}"
    )

    print(
        f"Similarity threshold: "
        f"{SIMILARITY_THRESHOLD}"
    )

    print(
        f"Grounding threshold: "
        f"{GROUNDING_THRESHOLD}"
    )

    print(
        "\n" + "=" * 70
    )

    query = input(
        "Ask a question: "
    ).strip()

    if not query:

        print(
            "Please enter a question."
        )

        return

    print(
        "\nSearching course knowledge base..."
    )

    result = answer_question(
        query,
        df
    )

    results = result["retrieved"]
    
        # --------------------------------------------------------
    # OUT-OF-SCOPE QUESTION
    # --------------------------------------------------------
    # If the question does not belong to the course,
    # do not expose irrelevant retrieved videos.
    # Retrieval is still performed internally, but the
    # user only sees the final "not covered" response.
    # --------------------------------------------------------

    if result["answer_method"] == "Not covered":

        print(
            "\n" + "=" * 70
        )

        print("ANSWER")

        print(
            "=" * 70
        )

        print(
            result["answer"]
        )

        print(
            "\n" + "=" * 70
        )

        return

    # --------------------------------------------------------
    # RETRIEVED SOURCES
    # --------------------------------------------------------

    print(
        "\nTop retrieved sources:"
    )

    strongest = (
        results
        .sort_values(
            "similarity",
            ascending=False
        )
        .head(TOP_K)
    )

    for _, row in strongest.iterrows():

        print(
            f"- Video {row['number']} | "
            f"{row['title']} | "
            f"{row['start']:.2f}s - "
            f"{row['end']:.2f}s | "
            f"similarity="
            f"{row['similarity']:.4f}"
        )

    print(
        "\nGenerating grounded answer..."
    )

    print(
        "\n" + "=" * 70
    )

    print("ANSWER")

    print(
        "=" * 70
    )

    print(
        result["answer"]
    )

    # --------------------------------------------------------
    # SOURCE
    # --------------------------------------------------------

    source = result["source"]

    if source is not None:

        print(
            "\n" + "-" * 70
        )

        print("SOURCE")

        print(
            "-" * 70
        )

        print(
            f"Video {source['number']}: "
            f"{source['title']}"
        )

        print(
            f"Timestamp: "
            f"{source['start']:.2f}s - "
            f"{source['end']:.2f}s"
        )

        print(
            f"Retrieval similarity: "
            f"{result['similarity']:.4f}"
        )

        print(
            f"Answer method: "
            f"{result['answer_method']}"
        )

        print(
            f"Grounding score: "
            f"{result['grounding_score']:.4f}"
        )

    print(
        "\n" + "=" * 70
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()