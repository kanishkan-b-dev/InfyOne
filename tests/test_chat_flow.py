"""
Behaviour tests for the rebuilt LangChain + LangGraph chat flow.
Tests the 7 required scenarios from Step 5:
1. "hello" -> friendly greeting, no retrieval
2. "what can you do?" -> capability reply, no retrieval
3. "who is the next CEO of Infosys?" -> retrieval, answer mentions Ashiss Kumar Dash, with a source
4. then "when does he take over?" -> follow-up, rewritten to standalone query, correct answer (April 2027)
5. "what was my first question?" -> answers with exact first message, no retrieval
6. "what's the capital of France?" -> out_of_scope reply
7. Question not in the data -> says not in knowledge base, no hallucination
"""
import asyncio
import sys
import uuid
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import pytest
from dotenv import load_dotenv

load_dotenv(ROOT_DIR / ".env")

from app.chatbot.graph import run_chat


@pytest.mark.asyncio
async def test_01_hello_greeting():
    session_id = f"test_{uuid.uuid4().hex[:8]}"
    res = await run_chat(session_id=session_id, question="hello")
    print("\n[Test 1: hello]")
    print("Intent:", res["intent"])
    print("Answering Node:", res["answering_node"])
    print("Answer:", res["answer"])
    assert res["intent"] == "greeting_smalltalk"
    assert res["answering_node"] == "respond_smalltalk"
    assert len(res["sources"]) == 0
    assert len(res["answer"]) > 10


@pytest.mark.asyncio
async def test_02_what_can_you_do():
    session_id = f"test_{uuid.uuid4().hex[:8]}"
    res = await run_chat(session_id=session_id, question="what can you do?")
    print("\n[Test 2: what can you do?]")
    print("Intent:", res["intent"])
    print("Answering Node:", res["answering_node"])
    print("Answer:", res["answer"])
    assert res["intent"] == "meta_bot"
    assert res["answering_node"] == "respond_meta_bot"
    assert not res["answer"].startswith("Hello!")
    assert len(res["sources"]) == 0


@pytest.mark.asyncio
async def test_03_04_05_multi_turn_flow():
    session_id = f"test_flow_{uuid.uuid4().hex[:8]}"

    # Turn 1: "who is the next CEO of Infosys?"
    res1 = await run_chat(session_id=session_id, question="who is the next CEO of Infosys?")
    print("\n[Test 3: who is the next CEO of Infosys?]")
    print("Intent:", res1["intent"])
    print("Answering Node:", res1["answering_node"])
    print("Answer:", res1["answer"])
    print("Sources:", [s["title"] for s in res1["sources"]])
    assert res1["intent"] == "infosys_question"
    assert res1["answering_node"] == "generate_answer"
    assert "Ashiss" in res1["answer"] or "Dash" in res1["answer"]
    assert len(res1["sources"]) > 0

    # Turn 2: "when does he take over?" (follow-up)
    res2 = await run_chat(session_id=session_id, question="when does he take over?")
    print("\n[Test 4: when does he take over?]")
    print("Intent:", res2["intent"])
    print("Rewritten Query:", res2["rewritten_query"])
    print("Answering Node:", res2["answering_node"])
    print("Answer:", res2["answer"])
    assert res2["intent"] == "follow_up"
    assert res2["answering_node"] == "generate_answer"
    assert res2["rewritten_query"] is not None
    assert "2027" in res2["answer"] or "April" in res2["answer"]

    # Turn 3: "what was my first question?" (chat history)
    res3 = await run_chat(session_id=session_id, question="what was my first question?")
    print("\n[Test 5: what was my first question?]")
    print("Intent:", res3["intent"])
    print("Answering Node:", res3["answering_node"])
    print("Answer:", res3["answer"])
    assert res3["intent"] == "chat_history"
    assert res3["answering_node"] == "respond_from_history"
    assert "who is the next CEO of Infosys?" in res3["answer"]
    assert len(res3["sources"]) == 0


@pytest.mark.asyncio
async def test_06_out_of_scope():
    session_id = f"test_{uuid.uuid4().hex[:8]}"
    res = await run_chat(session_id=session_id, question="what's the capital of France?")
    print("\n[Test 6: what's the capital of France?]")
    print("Intent:", res["intent"])
    print("Answering Node:", res["answering_node"])
    print("Answer:", res["answer"])
    assert res["intent"] == "out_of_scope"
    assert res["answering_node"] == "respond_out_of_scope"
    assert len(res["sources"]) == 0
    assert "Paris" not in res["answer"]  # polite refusal, does not answer French trivia


@pytest.mark.asyncio
async def test_07_question_not_in_data():
    session_id = f"test_{uuid.uuid4().hex[:8]}"
    res = await run_chat(session_id=session_id, question="What is Infosys's employee pet care allowance policy?")
    print("\n[Test 7: Question not in data]")
    print("Intent:", res["intent"])
    print("Answering Node:", res["answering_node"])
    print("Answer:", res["answer"])
    assert res["intent"] == "infosys_question"
    assert res["answering_node"] == "generate_answer"
    # Must declare not in knowledge base, no hallucination
    assert any(phrase in res["answer"].lower() for phrase in [
        "not in the", "could not find", "not available", "knowledge base", "no information"
    ])


if __name__ == "__main__":
    async def run_all():
        await test_01_hello_greeting()
        await test_02_what_can_you_do()
        await test_03_04_05_multi_turn_flow()
        await test_06_out_of_scope()
        await test_07_question_not_in_data()
        print("\nAll 7 behavioural tests passed successfully!")

    asyncio.run(run_all())
