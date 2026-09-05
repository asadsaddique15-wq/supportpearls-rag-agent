# The RAG chain (Task 07): composes retrieval, context formatting, prompting,
# the LLM call, and structured-output parsing into one callable.
# Task 10: also resolves citations to (source, location) pairs, de-duplicated.
from __future__ import annotations
import logging
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from src.config import settings
from src.chains.context import format_context
from src.chains.prompts import SYSTEM_PROMPT_V1
from src.chains.schemas import RAGResponse, Confidence, Citation
logger = logging.getLogger(__name__)

def get_llm():
    # Single source of truth for the chat model -- temperature must be low for grounded answers.
    # Uses the .env-configured key. For a user-supplied key (e.g. from the Streamlit UI),
    # build a ChatOpenAI instance directly and pass it into answer_question(llm=...) instead.
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key,
    )

def _fallback_response(reason: str) -> RAGResponse:
    # Used whenever something fails -- retrieval empty, API error, bad schema -- so the
    # caller always gets a valid RAGResponse object, never a crash or a raw traceback.
    return RAGResponse(
        answer=f"I'm sorry, I couldn't process that question right now ({reason}). Please contact support for help.",
        sources=[],
        confidence=Confidence.NONE,
        answered=False,
    )

def resolve_citations(response: RAGResponse, context) -> list[Citation]:
    # Task 10: turn the model's claimed S-labels into de-duplicated, display-ready citations
    # with source + location. Accepts either the exact label or a matching filename, since
    # models don't always follow the label format exactly.
    valid_labels = set(context.label_to_source.keys())
    valid_filenames = set(context.label_to_source.values())
    citations: list[Citation] = []
    seen: set[tuple[str, str]] = set()

    for claim in response.sources:
        source = None
        location = None
        if claim in valid_labels:
            source = context.label_to_source[claim]
            location = context.label_to_location.get(claim, "")
        elif claim in valid_filenames:
            # Model cited a filename directly -- find the matching label to get its location.
            for label, src in context.label_to_source.items():
                if src == claim:
                    source = src
                    location = context.label_to_location.get(label, "")
                    break
        if source is None:
            logger.warning("Model cited unknown source %r not in supplied context -- dropping.", claim)
            continue
        key = (source, location)
        if key not in seen:
            seen.add(key)
            citations.append(Citation(source=source, location=location))

    return citations

def answer_question(question: str, retriever, llm=None) -> tuple[RAGResponse, list[Citation]]:
    # Main entry point: takes a question and a configured retriever, returns a validated
    # RAGResponse plus a de-duplicated list of resolved Citations.
    # llm: optional pre-built ChatOpenAI instance, e.g. one using a user-supplied API key
    # from the Streamlit UI. Falls back to get_llm() (env-based) if not provided.
    # Step 1: retrieve
    try:
        chunks = retriever.invoke(question)
    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)
        return _fallback_response("retrieval error"), []
    # Step 2: relevance gate -- empty retrieval means an immediate, deterministic refusal.
    if not chunks:
        logger.info("Relevance gate triggered: no chunks retrieved for question=%r", question)
        response = RAGResponse(
            answer="I couldn't find anything in our documentation about that. Please contact support@pearlzhome.example for help.",
            sources=[],
            confidence=Confidence.NONE,
            answered=False,
        )
        return response, []
    # Step 3: format context with source labels
    context = format_context(chunks)
    if context.truncated:
        logger.warning("Context was truncated at %d chars for question=%r", len(context.text), question)
    # Step 4: build the prompt and call the LLM with structured output
    active_llm = llm or get_llm()
    structured_llm = active_llm.with_structured_output(RAGResponse)
    system_message = SYSTEM_PROMPT_V1.format(context=context.text)
    try:
        response: RAGResponse = structured_llm.invoke(
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": question},
            ]
        )
    except Exception as exc:
        logger.error("LLM call or schema validation failed: %s", exc)
        # Retry once before falling back, per Task 07's requirement.
        try:
            response = structured_llm.invoke(
                [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": question},
                ]
            )
        except Exception as retry_exc:
            logger.error("Retry also failed: %s", retry_exc)
            return _fallback_response("model error"), []

    # Step 5: resolve citations (Task 10) -- de-duplicated, source+location, refusal-safe.
    if response.answered:
        citations = resolve_citations(response, context)
        response.sources = [c.source for c in citations]
    else:
        # In the refusal case, there must be no citations shown, even if the model
        # tried to cite something -- a refusal with sources attached is misleading.
        citations = []
        response.sources = []

    logger.info(
        "Answered question=%r | answered=%s confidence=%s citations=%s",
        question, response.answered, response.confidence,
        [(c.source, c.location) for c in citations],
    )
    return response, citations