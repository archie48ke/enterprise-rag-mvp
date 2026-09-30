"""
AI Enterprise Knowledge Assistant — Streamlit MVP UI.

Run with:
    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import GROQ_API_KEY, INDEX_DIR, DOMAINS, DOMAIN_FOLDER
from src.workflow import run_workflow

st.set_page_config(page_title="AI Enterprise Knowledge Assistant", page_icon="🧠")


def indexes_exist() -> bool:
    for folder in DOMAIN_FOLDER.values():
        if (INDEX_DIR / folder / "index.faiss").exists():
            return True
    return False


def render_sidebar():
    with st.sidebar:
        st.title("AI Enterprise Knowledge Assistant")

        st.subheader("System Status")

        st.markdown("✓ Embedding Model")

        if indexes_exist():
            st.markdown("✓ FAISS Indexes")
        else:
            st.markdown("✗ FAISS Indexes")
            st.caption("Run: `python scripts/build_indexes.py`")

        if GROQ_API_KEY:
            st.markdown("✓ Groq Configuration")
        else:
            st.markdown("✗ Groq Configuration")
            st.caption("Set GROQ_API_KEY in your .env file")

        st.subheader("Domains")
        for d in DOMAINS:
            st.markdown(f"• {d.title() if d != 'HR' else 'HR'}")


def main():
    render_sidebar()

    st.title("AI Enterprise Knowledge Assistant")
    st.caption("Ask a question about company documents.")

    if not GROQ_API_KEY:
        st.error(
            "GROQ_API_KEY is not configured. Copy `.env.example` to `.env` "
            "and add your Groq API key before asking questions."
        )
        return

    if not indexes_exist():
        st.warning(
            "No FAISS indexes were found. Add PDFs under `data/<department>/` "
            "and run `python scripts/build_indexes.py` before asking questions."
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Replay chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question about company documents...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = run_workflow(question)
                except Exception as exc:
                    error_text = f"⚠️ An error occurred while processing your question: {exc}"
                    st.error(error_text)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_text}
                    )
                    return

            domain = result.get("domain", "GENERAL")
            answer = result.get("answer", "")
            citations = result.get("citations", [])

            response_parts = [f"**Detected Department:** {domain}"]
            response_parts.append(f"\n**Answer:**\n\n{answer}")

            if citations:
                sources_md = "\n".join(f"- {c.display()}" for c in citations)
                response_parts.append(f"\n**Sources:**\n\n{sources_md}")

            full_response = "\n".join(response_parts)
            st.markdown(full_response)

            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )


if __name__ == "__main__":
    main()
