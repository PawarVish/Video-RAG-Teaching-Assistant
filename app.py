import streamlit as st

from scripts.rag_assistant import (
    load_embeddings,
    answer_question
)

def format_timestamp(seconds):

    seconds = int(seconds)

    minutes = seconds // 60
    remaining_seconds = seconds % 60

    return f"{minutes:02d}:{remaining_seconds:02d}"

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Video RAG Teaching Assistant",
    page_icon="🎓",
    layout="centered"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1rem;
        opacity: 0.75;
        margin-bottom: 1.5rem;
    }

    .source-box {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,0.3);
        margin-top: 0.5rem;
    }

    .metric-label {
        font-size: 0.85rem;
        opacity: 0.7;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎓 Video RAG Teaching Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions about the course using semantic retrieval '
    'and a local LLM.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

@st.cache_resource
def load_knowledge_base():

    return load_embeddings()


with st.spinner("Loading course knowledge base..."):

    df = load_knowledge_base()


st.success(
    f"Knowledge base ready • {len(df):,} transcript chunks"
)


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.text_input(
    "Ask your question",
    placeholder="Example: What is VS Code?",
    label_visibility="visible"
)


# ============================================================
# ASK BUTTON
# ============================================================

ask_clicked = st.button(
    "Ask Question",
    type="primary",
    use_container_width=True
)


if ask_clicked:

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching course knowledge and generating answer..."
        ):

            result = answer_question(
                question.strip(),
                df
            )

        # ====================================================
        # ANSWER
        # ====================================================

        st.divider()

        st.subheader("Answer")

        st.write(
            result["answer"]
        )

        # ====================================================
        # OUT-OF-SCOPE
        # ====================================================

        if result["answer_method"] == "Not covered":

            st.info(
                "This question is outside the available "
                "course content."
            )

        # ====================================================
        # SOURCE
        # ====================================================

        else:

            source = result["source"]

            if source is not None:

                st.subheader("📚 Source")

                start_time = format_timestamp(
                    source["start"]
                )

                end_time = format_timestamp(
                    source["end"]
                )

                st.markdown(
                    f"""
                    <div class="source-box">

                    <b>Video {source['number']}</b><br>
                    {source['title']}<br><br>

                    <b>Timestamp</b><br>
                    {start_time} – {end_time}

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # --------------------------------------------
                # Technical details
                # --------------------------------------------

                with st.expander(
                    "Retrieval & grounding details"
                ):

                    col1, col2 = st.columns(2)

                    with col1:

                        st.metric(
                            "Retrieval similarity",
                            f"{result['similarity']:.4f}"
                        )

                    with col2:

                        st.metric(
                            "Grounding score",
                            f"{result['grounding_score']:.4f}"
                        )

                    st.caption(
                        f"Answer method: "
                        f"{result['answer_method']}"
                    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About this project")

    st.write(
        "A local Retrieval-Augmented Generation system "
        "that answers questions from timestamped course "
        "video transcripts."
    )

    st.divider()

    st.subheader("Pipeline")

    st.write(
        "🎥 Video → Audio"
    )

    st.write(
        "🎙️ Whisper → Transcript"
    )

    st.write(
        "🧩 Transcript → Chunks"
    )

    st.write(
        "🔎 BGE-M3 → Semantic Retrieval"
    )

    st.write(
        "🤖 Llama 3.2 1B → Answer"
    )

    st.write(
        "🛡️ Grounding Validation"
    )

    st.divider()

    st.subheader("Models")

    st.write(
        "**Embedding:** BGE-M3"
    )

    st.write(
        "**LLM:** Llama 3.2 1B"
    )

    st.write(
        "**Retrieval:** Cosine Similarity"
    )

    st.divider()

    st.caption(
        "Video RAG Teaching Assistant"
    )