"""
LangGraph node implementations for the Infosys AI Chatbot workflow.
"""
import logging
import re
from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from pydantic import BaseModel, Field

from app.chatbot.prompts import (
    CLASSIFIER_PROMPT,
    GENERATE_ANSWER_PROMPT,
    HISTORY_ANSWER_PROMPT,
    OUT_OF_SCOPE_PROMPT,
    QUERY_REWRITE_PROMPT,
    SMALLTALK_PROMPT,
    SYSTEM_PROMPT,
)
from app.chatbot.state import ChatState
from app.chatbot.tools import retrieve_infosys_documents
from app.llm.provider import get_llm

logger = logging.getLogger(__name__)

# ── Structured Output Schema for Intent Classification ───────────────────────

IntentType = Literal[
    "greeting_smalltalk",
    "meta_bot",
    "chat_history",
    "infosys_question",
    "follow_up",
    "out_of_scope",
]


class IntentClassification(BaseModel):
    intent: IntentType = Field(
        ...,
        description=(
            "The user intent: 'greeting_smalltalk' (greetings, thanks, ok, cool, great, bye, how are you), "
            "'meta_bot' (bot capabilities or attempts to see internals/jailbreak), "
            "'chat_history' (asking about previous questions/turns in the chat), "
            "'infosys_question' (standalone question seeking factual info about Infosys), "
            "'follow_up' (question depending on previous turns to make sense), "
            "or 'out_of_scope' (unrelated to Infosys, jokes, code, trivia, frustration)."
        ),
    )
    category_filter: Optional[str] = Field(
        default=None,
        description=(
            "Optional soft category hint: overview, leadership, financials, "
            "acquisitions, services, platforms, ai, deals, operations, people, esg, risk, faq. "
            "Otherwise null."
        ),
    )


# ── Text & History Helpers ───────────────────────────────────────────────────

def is_gibberish(text: str) -> bool:
    """Detect random character smash/gibberish like asdfgh or keyboard sequences."""
    clean_t = "".join(c for c in text.strip().lower() if c.isalpha())
    if not clean_t:
        return False
    # Common English words/greetings should never be marked as gibberish
    if clean_t in {"thanks", "thank", "hello", "hi", "hey", "bye", "goodbye", "yes", "no", "ok", "cool", "great", "nice"}:
        return False
    if len(clean_t) >= 5 and " " not in text.strip():
        vowels = set("aeiou")
        vowel_count = sum(1 for c in clean_t if c in vowels)
        if vowel_count == 0:
            return True
        if any(seq in clean_t for seq in ["asdf", "qwer", "zxcv", "ghjk", "dfgh", "hjkl"]):
            return True
        if re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", clean_t):
            return True
    return False


def is_frustration(text: str) -> bool:
    """Detect user frustration or complaints like 'you are useless' or 'this is bad'."""
    t = text.lower().strip()
    frustration_phrases = [
        "useless", "this is bad", "you are bad", "terrible", "awful",
        "worst", "hate you", "stupid", "idiot", "nonsense", "not helpful", "garbage"
    ]
    return any(p in t for p in frustration_phrases)


def _trim_history(messages: List[BaseMessage], max_messages: int = 20) -> List[BaseMessage]:
    """Keep the last N messages to prevent context overflow while preserving flow."""
    if len(messages) <= max_messages:
        return messages
    return messages[-max_messages:]


def _format_messages_as_text(messages: List[BaseMessage]) -> str:
    """Format message list into a clean readable transcript."""
    lines = []
    for msg in messages:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def _get_last_user_message(messages: List[BaseMessage]) -> str:
    """Extract the most recent user message string."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


def _filter_used_sources(documents: List[Dict[str, Any]], answer: str, max_sources: int = 3) -> List[Dict[str, Any]]:
    """
    Select up to 3 de-duplicated sources actually used in the answer.
    Never show FAQ chunks as the cited source.
    """
    used_sources = []
    seen_titles = set()
    answer_lower = answer.lower()

    # Pass 1: Chunks whose title is cited or referenced in the answer
    for doc in documents:
        meta = doc.get("metadata", {})
        title = meta.get("title", "Infosys Document")
        cat = str(meta.get("category", "")).lower()
        src = str(meta.get("source", "")).lower()

        if cat == "faq" or src == "curated-faq":
            continue

        norm_title = title.strip().lower()
        if norm_title in seen_titles:
            continue

        # Check if title or key title words appear in the answer
        title_words = [w for w in re.findall(r"\w+", norm_title) if len(w) > 4]
        if norm_title in answer_lower or (title_words and any(w in answer_lower for w in title_words)):
            seen_titles.add(norm_title)
            used_sources.append({
                "title": title,
                "source": meta.get("source", "https://www.infosys.com"),
                "category": cat,
                "as_of": meta.get("as_of", "N/A"),
            })
            if len(used_sources) >= max_sources:
                return used_sources

    # Pass 2: Fill remaining slots from top non-FAQ retrieved documents
    for doc in documents:
        if len(used_sources) >= max_sources:
            break
        meta = doc.get("metadata", {})
        title = meta.get("title", "Infosys Document")
        cat = str(meta.get("category", "")).lower()
        src = str(meta.get("source", "")).lower()

        if cat == "faq" or src == "curated-faq":
            continue

        norm_title = title.strip().lower()
        if norm_title not in seen_titles:
            seen_titles.add(norm_title)
            used_sources.append({
                "title": title,
                "source": meta.get("source", "https://www.infosys.com"),
                "category": cat,
                "as_of": meta.get("as_of", "N/A"),
            })

    return used_sources[:max_sources]


# ── Node 0: reset_turn_state ─────────────────────────────────────────────────

def reset_turn_state(state: ChatState) -> Dict[str, Any]:
    """
    State hygiene: Reset transient per-turn variables so they do not leak from previous turns.
    """
    return {
        "intent": "",
        "rewritten_query": "",
        "category_filter": None,
        "documents": [],
        "sources": [],
        "answer": "",
        "answering_node": "",
    }


# ── Node 1: classify_intent ──────────────────────────────────────────────────

async def classify_intent(state: ChatState) -> Dict[str, Any]:
    """
    Classify intent into one of 6 intents with category soft hint.
    """
    messages = state.get("messages", [])
    recent_context = _trim_history(messages, max_messages=8)
    last_user_msg = _get_last_user_message(messages).strip()

    # Fast heuristic check for gibberish
    if is_gibberish(last_user_msg):
        logger.info("[classify_intent] Fast-path detected gibberish -> out_of_scope")
        return {
            "intent": "out_of_scope",
            "category_filter": None,
        }

    # Fast heuristic check for frustration
    if is_frustration(last_user_msg):
        logger.info("[classify_intent] Fast-path detected frustration -> out_of_scope")
        return {
            "intent": "out_of_scope",
            "category_filter": None,
        }

    # Fast heuristic check for common smalltalk / thanks / ok / cool / great / nice
    q_clean = "".join(c for c in last_user_msg.lower() if c.isalnum() or c.isspace()).strip()
    if q_clean in {"hi", "hello", "hey", "good morning", "good evening", "how are you", "how are you doing",
                   "thanks", "thank you", "thx", "ok", "cool", "great", "nice", "bye", "goodbye", "see you"}:
        logger.info("[classify_intent] Fast-path detected greeting_smalltalk: '%s'", q_clean)
        return {
            "intent": "greeting_smalltalk",
            "category_filter": None,
        }

    llm = get_llm()
    classifier = llm.with_structured_output(IntentClassification)

    context_transcript = _format_messages_as_text(recent_context)

    prompt_messages = [
        SystemMessage(content=CLASSIFIER_PROMPT),
        HumanMessage(
            content=(
                f"Conversation History (most recent last):\n{context_transcript}\n\n"
                f"Latest User Message: {last_user_msg}\n\n"
                f"Classify the latest user message into one of the 6 intents."
            )
        ),
    ]

    try:
        result = await classifier.ainvoke(prompt_messages)
        intent = result.intent
        cat_filter = result.category_filter
    except Exception as exc:
        logger.warning("[classify_intent] Classifier error: %s; falling back to infosys_question", exc)
        intent = "infosys_question"
        cat_filter = None

    logger.info("[classify_intent] Classified message as intent='%s', category_filter=%s", intent, cat_filter)

    return {
        "intent": intent,
        "category_filter": cat_filter,
    }


# ── Node 2: respond_smalltalk ────────────────────────────────────────────────

async def respond_smalltalk(state: ChatState) -> Dict[str, Any]:
    """
    Reply directly and briefly to greetings/pleasantries matching user message:
    - greeting -> short greeting (only greetings may start with 'Hello!'). Offer 1-2 examples.
    - 'how are you' -> one short friendly line. Offer 1-2 examples.
    - thanks -> 'You're welcome!'. Offer 1-2 examples.
    - bye -> short goodbye with NO re-introduction and NO example questions.
    NO retrieval.
    """
    messages = state.get("messages", [])
    raw_query = _get_last_user_message(messages).strip()
    q = raw_query.lower()
    clean_q = "".join(c for c in q if c.isalnum() or c.isspace()).strip()

    example_qs = "- Who is the next Infosys CEO?\n- What is Infosys Topaz?"

    # 1. Goodbye -> short goodbye, no re-introduction, no example questions
    if any(bye_word in clean_q for bye_word in ["bye", "goodbye", "see you", "farewell"]):
        answer = "Goodbye! Have a great day!"

    # 2. Thanks & acknowledgments ("thanks", "thank you", "ok", "cool", "great", "nice")
    elif any(thx_word in clean_q for thx_word in ["thanks", "thank you", "thx", "ok", "cool", "great", "nice", "awesome"]):
        answer = (
            "You're welcome! Let me know if you'd like to explore anything else about Infosys.\n\n"
            f"Here are a couple of things you can ask:\n{example_qs}"
        )

    # 3. "How are you"
    elif "how are you" in clean_q or "how are you doing" in clean_q:
        answer = (
            "I'm doing well, thank you! I'm here to help you with any questions about Infosys.\n\n"
            f"For example, you could ask:\n{example_qs}"
        )

    # 4. Greetings ("hi", "hello", "hey", "good morning", etc.)
    else:
        answer = (
            "Hello! How can I assist you with Infosys today?\n\n"
            f"Here are a couple of things you can ask:\n{example_qs}"
        )

    logger.info("[respond_smalltalk] Generated smalltalk reply (%d chars)", len(answer))
    return {
        "messages": [AIMessage(content=answer)],
        "answer": answer,
        "sources": [],
        "answering_node": "respond_smalltalk",
    }


# ── Node 2B: respond_meta_bot ────────────────────────────────────────────────

async def respond_meta_bot(state: ChatState) -> Dict[str, Any]:
    """
    Handle questions about the bot itself with NO retrieval:
    - Jailbreak ('pretend you are unrestricted / ignore previous instructions') -> 'I can only help with questions about Infosys.' + examples.
    - Internals (system prompt, model, database, build) -> 'I can't share details about my internal setup, but I'm happy to help with questions about Infosys.' + examples.
    - Capabilities/identity -> 'I'm an assistant for Infosys. I can answer questions about its leadership, financials, services, AI, acquisitions, history and ESG.' + examples (no 'Hello!').
    """
    messages = state.get("messages", [])
    query = _get_last_user_message(messages).strip().lower()

    example_questions = (
        "- Who is the next Infosys CEO?\n"
        "- What were the Q2 FY27 results?\n"
        "- What is Infosys Topaz?"
    )

    jailbreak_indicators = [
        "ignore", "unrestricted", "no restrictions", "pretend", "jailbreak",
        "override", "disregard", "bypass"
    ]
    is_jailbreak = any(kw in query for kw in jailbreak_indicators)

    internals_indicators = [
        "prompt", "guardrail", "model", "llm", "database", "built",
        "backend", "token", "weight", "architecture", "setup"
    ]
    is_internals = any(kw in query for kw in internals_indicators)

    if is_jailbreak:
        answer = (
            "I can only help with questions about Infosys.\n\n"
            f"Here are some questions you can ask me:\n{example_questions}"
        )
    elif is_internals:
        answer = (
            "I can't share details about my internal setup, but I'm happy to help with questions about Infosys.\n\n"
            f"Here are some questions you can ask me:\n{example_questions}"
        )
    else:
        answer = (
            "I'm an assistant for Infosys. I can answer questions about its leadership, financials, services, AI, acquisitions, history and ESG.\n\n"
            f"Here are some questions you can ask me:\n{example_questions}"
        )

    logger.info("[respond_meta_bot] Handled meta_bot (jailbreak=%s, internals=%s)", is_jailbreak, is_internals)
    return {
        "messages": [AIMessage(content=answer)],
        "answer": answer,
        "sources": [],
        "answering_node": "respond_meta_bot",
    }


# ── Node 3: respond_from_history ─────────────────────────────────────────────

async def respond_from_history(state: ChatState) -> Dict[str, Any]:
    """
    Answer chat-history questions computed in code (not by LLM) for counts, first question,
    and last question. Only uses LLM for summarization.
    NO retrieval.
    """
    messages = state.get("messages", [])
    last_user_msg = _get_last_user_message(messages).strip()
    query_lower = last_user_msg.lower()

    human_msgs = [m for m in messages if isinstance(m, HumanMessage) or getattr(m, "type", "") == "human"]
    ai_msgs = [m for m in messages if isinstance(m, AIMessage) or getattr(m, "type", "") == "ai"]

    earlier_human_msgs = human_msgs[:-1] if len(human_msgs) > 1 else []

    # 1. First question
    if any(k in query_lower for k in ["first question", "first query"]):
        if not earlier_human_msgs:
            answer = "You haven't asked any previous questions yet."
        else:
            first_q = earlier_human_msgs[0].content
            answer = f'Your first question was: "{first_q}"'

    # 2. Last question (must exclude current question)
    elif any(k in query_lower for k in ["last question", "previous question", "what did i just ask"]):
        if not earlier_human_msgs:
            answer = "You haven't asked any previous questions yet."
        else:
            last_q = earlier_human_msgs[-1].content
            answer = f'Your last question was: "{last_q}"'

    # 3. How many questions / count
    elif any(k in query_lower for k in ["how many questions", "how many queries", "count of questions", "number of questions"]):
        prior_count = len(earlier_human_msgs)
        total_count = len(human_msgs)
        if prior_count == 0:
            answer = "You haven't asked any previous questions yet (this is your first question)."
        else:
            plural = "questions" if prior_count != 1 else "question"
            answer = f"You have asked {prior_count} previous {plural} so far (this question makes it {total_count})."

    # 4. What did you just say / repeat
    elif any(k in query_lower for k in ["what did you just say", "what did you say", "repeat that", "repeat your last"]):
        if not ai_msgs:
            answer = "I haven't said anything yet in this conversation."
        else:
            last_ai = ai_msgs[-1].content
            answer = f'I said: "{last_ai}"'

    # 5. Summarize our conversation
    elif any(k in query_lower for k in ["summarize", "summary", "recap"]):
        if not earlier_human_msgs:
            answer = "There is no previous conversation to summarize yet."
        else:
            history_text = _format_messages_as_text(_trim_history(messages[:-1], max_messages=20))
            llm = get_llm()
            prompt = (
                f"You are an assistant for Infosys. Summarize our conversation so far.\n\n"
                f"CONVERSATION HISTORY:\n{history_text}\n\n"
                "Provide a concise summary of the questions asked and answers provided."
            )
            resp = await llm.ainvoke([HumanMessage(content=prompt)])
            answer = str(resp.content).strip()

    # Fallback to LLM
    else:
        history_text = _format_messages_as_text(_trim_history(messages[:-1], max_messages=20))
        llm = get_llm()
        resp = await llm.ainvoke([
            SystemMessage(content=HISTORY_ANSWER_PROMPT),
            HumanMessage(content=f"History:\n{history_text}\n\nUser asked: {last_user_msg}")
        ])
        answer = str(resp.content).strip()

    logger.info("[respond_from_history] Generated history answer (%d chars)", len(answer))
    return {
        "messages": [AIMessage(content=answer)],
        "answer": answer,
        "sources": [],
        "answering_node": "respond_from_history",
    }


# ── Node 4: rewrite_query ────────────────────────────────────────────────────

async def rewrite_query(state: ChatState) -> Dict[str, Any]:
    """
    Turn follow-ups into a standalone search query using the chat history.
    """
    messages = state.get("messages", [])
    recent_context = _trim_history(messages, max_messages=8)
    last_user_msg = _get_last_user_message(messages)

    context_transcript = _format_messages_as_text(recent_context[:-1])

    llm = get_llm()
    prompt_messages = [
        SystemMessage(content=QUERY_REWRITE_PROMPT),
        HumanMessage(
            content=(
                f"Conversation History:\n{context_transcript}\n\n"
                f"Follow-up Question: {last_user_msg}\n\n"
                f"Output the standalone search query:"
            )
        ),
    ]

    response = await llm.ainvoke(prompt_messages)
    rewritten = str(response.content).strip().strip('"').strip("'")

    logger.info("[rewrite_query] Follow-up '%s' -> Standalone '%s'", last_user_msg, rewritten)
    return {
        "rewritten_query": rewritten,
    }


# ── Node 5: retrieve ─────────────────────────────────────────────────────────

async def retrieve(state: ChatState) -> Dict[str, Any]:
    """
    Retrieve top_k=8 documents across the collection without hard category filter,
    applying optional category soft re-ranking boost.
    """
    query = state.get("rewritten_query") or _get_last_user_message(state.get("messages", []))
    category = state.get("category_filter")

    documents, _ = retrieve_infosys_documents(query=query, category=category, top_k=8)

    return {
        "documents": documents,
    }


# ── Node 6: generate_answer ──────────────────────────────────────────────────

async def generate_answer(state: ChatState) -> Dict[str, Any]:
    """
    Answer using retrieved documents + chat history.
    Filter returned sources to max 3 non-FAQ sources actually used in answer.
    """
    documents = state.get("documents", [])
    messages = state.get("messages", [])
    question = state.get("rewritten_query") or _get_last_user_message(messages)
    cat_filter = state.get("category_filter")

    async def _invoke_llm(docs: List[Dict[str, Any]]) -> str:
        context_blocks = []
        for i, doc in enumerate(docs, 1):
            meta = doc.get("metadata", {})
            title = meta.get("title", "Infosys Document")
            cat = meta.get("category", "")
            source = meta.get("source", "https://www.infosys.com")
            as_of = meta.get("as_of", "N/A")
            content = doc.get("content", "")

            context_blocks.append(
                f"[Document {i}]\n"
                f"Title: {title}\n"
                f"Category: {cat}\n"
                f"Source: {source}\n"
                f"As of: {as_of}\n"
                f"Content: {content}"
            )

        context_str = "\n\n---\n\n".join(context_blocks) if context_blocks else "(No matching documents found)"
        history_transcript = _format_messages_as_text(_trim_history(messages[:-1], max_messages=12))

        llm = get_llm()
        system_filled = GENERATE_ANSWER_PROMPT.format(
            context=context_str,
            history=history_transcript or "(No prior messages)",
            question=question,
        )

        prompt_messages = [
            SystemMessage(content=system_filled),
            HumanMessage(content=question),
        ]
        response = await llm.ainvoke(prompt_messages)
        return str(response.content).strip()

    answer = await _invoke_llm(documents)

    # If the answer returns "not found" while a filter was used, retry once without it
    is_not_found = any(phrase in answer.lower() for phrase in [
        "could not find that information in the infosys knowledge base",
        "not available in the infosys knowledge base",
    ])

    if is_not_found and cat_filter:
        logger.info(
            "[generate_answer] Answer was 'not found' while category_filter='%s'. Retrying retrieval without filter.",
            cat_filter,
        )
        retry_docs, _ = retrieve_infosys_documents(query=question, category=None, top_k=8)
        if retry_docs:
            retry_answer = await _invoke_llm(retry_docs)
            if not any(p in retry_answer.lower() for p in ["could not find that information", "not in the infosys knowledge base"]):
                answer = retry_answer
                documents = retry_docs

    # Filter sources to max 3 non-FAQ sources actually used in the answer
    final_sources = _filter_used_sources(documents, answer, max_sources=3)

    logger.info("[generate_answer] Generated answer (%d chars) | %d sources", len(answer), len(final_sources))
    return {
        "messages": [AIMessage(content=answer)],
        "answer": answer,
        "sources": final_sources,
        "answering_node": "generate_answer",
    }


# ── Node 7: respond_out_of_scope ─────────────────────────────────────────────

async def respond_out_of_scope(state: ChatState) -> Dict[str, Any]:
    """
    Politely say you only answer about Infosys.
    Handles frustration with empathetic reply.
    Handles gibberish with 'I didn't catch that. Could you rephrase?'.
    Suggests ONLY verified questions the knowledge base can answer.
    """
    messages = state.get("messages", [])
    last_user_msg = _get_last_user_message(messages).strip()

    example_questions = (
        "- Who is the next Infosys CEO?\n"
        "- What were the Q2 FY27 results?\n"
        "- What is Infosys Topaz?\n"
        "- Which companies has Infosys acquired?"
    )

    if is_frustration(last_user_msg):
        answer = (
            "I'm sorry to hear that. I'm here to provide accurate facts about Infosys and would love to help get you what you need.\n\n"
            f"Here are some questions you can ask me:\n{example_questions}"
        )
    elif is_gibberish(last_user_msg):
        answer = (
            "I didn't catch that. Could you rephrase?\n\n"
            f"Here are some questions you can ask me about Infosys:\n{example_questions}"
        )
    else:
        answer = (
            "Thank you for reaching out! My expertise is focused specifically on Infosys "
            "(its leadership, financials, services, AI platforms, acquisitions, history and ESG).\n\n"
            f"Here are some questions you can ask me about Infosys:\n{example_questions}"
        )

    logger.info("[respond_out_of_scope] Generated reply (%d chars)", len(answer))
    return {
        "messages": [AIMessage(content=answer)],
        "answer": answer,
        "sources": [],
        "answering_node": "respond_out_of_scope",
    }
