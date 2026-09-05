# Conversational memory (Task 09): condenses follow-up questions into standalone queries,
# and bounds chat history so it doesn't grow forever.
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from src.config import settings
from src.chains.rag_chain import get_llm
logger = logging.getLogger(__name__)
MAX_HISTORY_TURNS = 6  # keep the last N turns; older turns are dropped (windowing policy)
CONDENSE_PROMPT = """Given the conversation history and a follow-up question, rewrite the follow-up into a standalone question.
Rules:
1. If the follow-up depends on a pronoun or implicit reference (e.g. "is that covered?", "how much is it?"), resolve the reference by identifying exactly what topic/subject the previous turn was about, and preserve that subject precisely in the rewrite. Do not generalise or change the topic (e.g. if the previous turn was about the WARRANTY, the rewrite must still be asking about the WARRANTY, not a different but related topic like installation service).
2. If the follow-up introduces a NEW topic that is not a continuation of the previous turn, return the follow-up EXACTLY AS WRITTEN. Do not add any product name, entity, or detail from the history that the follow-up itself did not mention.
3. Only carry over an entity from history if the follow-up question would be unanswerable or ambiguous without it.
4. When in doubt about whether this is a topic switch, prefer returning the question unchanged over adding unrelated context.
Conversation history:
{history}
Follow-up question: {question}
Standalone question (respond with ONLY the rewritten question, nothing else):"""

@dataclass
class ChatSession:
    # Holds one customer's conversation state: bounded turn history.
    turns: list[tuple[str, str]] = field(default_factory=list)  # (user_question, assistant_answer)

    def add_turn(self, question: str, answer: str) -> None:
        self.turns.append((question, answer))
        if len(self.turns) > MAX_HISTORY_TURNS:
            self.turns = self.turns[-MAX_HISTORY_TURNS:]  # window: drop oldest turns
            
    def reset(self) -> None:
        self.turns = []

    def format_history(self) -> str:
        if not self.turns:
            return "(no previous turns)"
        lines = []
        for q, a in self.turns:
            lines.append(f"Customer: {q}")
            lines.append(f"Assistant: {a}")
        return "\n".join(lines)

def condense_query(session: ChatSession, question: str, llm=None) -> str:
    # llm: optional pre-built ChatOpenAI instance (e.g. using a user-supplied API key
    # from the Streamlit UI). Falls back to get_llm() (env-based) if not provided.
    # Rewrite the latest question into a standalone form using history. Logs both
    # original and rewritten query -- required for the Task 09 acceptance criterion.
    if not session.turns:
        logger.info("No history yet -- using question as-is: %r", question)
        return question
    active_llm = llm or get_llm()
    prompt = CONDENSE_PROMPT.format(history=session.format_history(), question=question)
    response = active_llm.invoke(prompt)
    standalone = response.content.strip()

    logger.info("Query condensation | original=%r | rewritten=%r", question, standalone)
    return standalone