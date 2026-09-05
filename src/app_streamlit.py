# Streamlit web UI for SupportPearlz. Requires the user to enter their own OpenAI
# API key before any access to the chat is granted -- no fallback to a stored key.

from __future__ import annotations
import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import AuthenticationError
from src.config import settings
from src.retrieval.retriever import get_similarity_retriever
from src.chains.rag_chain import answer_question
from src.chains.memory import ChatSession, condense_query
from src.utils.logging_setup import configure_logging

configure_logging()

st.set_page_config(page_title="SupportPearlz", page_icon="💧")


def validate_api_key(api_key: str) -> bool:
    # Makes one cheap real call to confirm the key actually works before granting access.
    try:
        test_embeddings = OpenAIEmbeddings(model=settings.embedding_model, api_key=api_key)
        test_embeddings.embed_query("connection test")
        return True
    except AuthenticationError:
        return False
    except Exception:
        return False


def build_session_resources(api_key: str):
    # Everything downstream uses THIS key, never settings.openai_api_key from .env.
    embeddings = OpenAIEmbeddings(model=settings.embedding_model, api_key=api_key)
    llm = ChatOpenAI(model=settings.llm_model, temperature=settings.llm_temperature, api_key=api_key)
    store = Chroma(
        collection_name=settings.vector_store_collection,
        embedding_function=embeddings,
        persist_directory=str(settings.vector_store_path),
    )
    retriever = get_similarity_retriever(store, k=settings.retrieval_k)
    return llm, retriever


# ---- Access gate ----
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("💧 SupportPearlz")
    st.markdown("### Access Required")
    st.write("Enter your OpenAI API key to use this assistant. Your key is used only for this session and is never stored.")

    api_key_input = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")

    if st.button("Connect"):
        if not api_key_input.strip():
            st.error("Please enter an API key.")
        else:
            with st.spinner("Validating key..."):
                is_valid = validate_api_key(api_key_input.strip())
            if is_valid:
                st.session_state.authenticated = True
                st.session_state.api_key = api_key_input.strip()
                st.session_state.llm, st.session_state.retriever = build_session_resources(st.session_state.api_key)
                st.session_state.chat_session = ChatSession()
                st.session_state.display_history = []
                st.rerun()
            else:
                st.error("Invalid API key. Access denied.")

    st.stop()  # Hard stop -- nothing below this runs until authenticated.

# ---- Main chat UI (only reachable after successful key validation) ----
st.title("💧 SupportPearlz")
st.caption("Ask about warranty, returns, shipping, installation, or troubleshooting.")

with st.sidebar:
    st.success("Connected")
    if st.button("Reset conversation"):
        st.session_state.chat_session.reset()
        st.session_state.display_history = []
        st.rerun()
    if st.button("Log out (clear API key)"):
        for key in ["authenticated", "api_key", "llm", "retriever", "chat_session", "display_history"]:
            st.session_state.pop(key, None)
        st.rerun()

for turn in st.session_state.display_history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])
        if turn.get("sources"):
            st.caption("Sources: " + ", ".join(turn["sources"]))
        if turn.get("rewritten"):
            st.caption(f"Rewritten query: {turn['rewritten']!r}")

question = st.chat_input("Type your question...")

if question:
    st.session_state.display_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            standalone = condense_query(
                st.session_state.chat_session, question, llm=st.session_state.llm
            )
            result, citations = answer_question(
                standalone, st.session_state.retriever, llm=st.session_state.llm
            )

        st.write(result.answer)
        source_labels = [f"{c.source} ({c.location})" for c in citations]
        if source_labels:
            st.caption("Sources: " + ", ".join(source_labels))
        st.caption(f"Confidence: {result.confidence.value}")
        if standalone != question:
            st.caption(f"Rewritten query: {standalone!r}")

        st.session_state.chat_session.add_turn(question, result.answer)
        st.session_state.display_history.append({
            "role": "assistant",
            "content": result.answer,
            "sources": source_labels,
            "rewritten": standalone if standalone != question else None,
        })