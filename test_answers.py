#!/usr/bin/env python
"""
Athena AI Chatbot — V2 Architecture Benchmark Runner
---------------------------------------------------
Runs the 100 benchmark questions (or subsets) through the V2 Architecture:
- Vector Store : AthenaCollegeCollection_v2 (Chroma Cloud)
- Embedding    : OpenAI text-embedding-3-large (3072-dim)
- Retrieval    : Multi-stage Candidate Retrieval + Reranking (RetrieverV2)
- Generation   : Strict Grounded Generation (GroundedGeneratorV2)

Outputs:
  - test_results_v2.txt
  - test_results_v2.json

Usage:
  python test_answers_v2.py                  # Runs all 100 questions
  python test_answers_v2.py --limit 5        # Runs first 5 questions
  python test_answers_v2.py --questions 1,2  # Runs specific question IDs
  python test_answers_v2.py --section A      # Runs section A questions
  python test_answers_v2.py --ask "My query" # Ask an interactive single question
"""

import argparse
import asyncio
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

# 1. Setup project paths and environment
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

# Explicitly load .env file from project root
from dotenv import load_dotenv
env_path = ROOT_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# Verify credentials before running
if not os.getenv("OPENAI_API_KEY") and not os.getenv("AZURE_OPENAI_API_KEY"):
    print("[ERROR] No OpenAI API Key found in .env file.")
    print(f"Please check your .env file at: {env_path}")
    sys.exit(1)

from app.services.retriever_service import get_retriever_v2
from app.services.grounded_service import get_grounded_generator_v2

# 2. Complete 100 Questions List
QUESTIONS = [
    # ── A. General Information & Campus Location (1–15) ──────────────────────
    {"id": 1, "section": "A", "question": "Where is Athena Institute of Technology located?"},
    {"id": 2, "section": "A", "question": "What are the standard administrative hours?"},
    {"id": 3, "section": "A", "question": "What are the academic lecture hours?"},
    {"id": 4, "section": "A", "question": "Is the college open on Sundays?"},
    {"id": 5, "section": "A", "question": "What are the Saturday administrative hours?"},
    {"id": 6, "section": "A", "question": "How can I reach the campus using public transportation?"},
    {"id": 7, "section": "A", "question": "Which metro station is closest to the campus?"},
    {"id": 8, "section": "A", "question": "Which bus routes stop at the college?"},
    {"id": 9, "section": "A", "question": "Does the college provide a shuttle service?"},
    {"id": 10, "section": "A", "question": "What time does the campus shuttle operate?"},
    {"id": 11, "section": "A", "question": "What is the official address of AIT?"},
    {"id": 12, "section": "A", "question": "What are the main campus facilities?"},
    {"id": 13, "section": "A", "question": "Does AIT provide parking facilities?"},
    {"id": 14, "section": "A", "question": "Is there an EV charging facility on campus?"},
    {"id": 15, "section": "A", "question": "Where can students find bicycle parking?"},

    # ── B. Undergraduate Programs (16–25) ────────────────────────────────────
    {"id": 16, "section": "B", "question": "What B.Tech programs does AIT offer?"},
    {"id": 17, "section": "B", "question": "What undergraduate engineering programs are available?"},
    {"id": 18, "section": "B", "question": "What are all the B.Tech branches at AIT?"},
    {"id": 19, "section": "B", "question": "How long are the B.Tech programs?"},
    {"id": 20, "section": "B", "question": "Does AIT offer Computer Science Engineering?"},
    {"id": 21, "section": "B", "question": "Does AIT offer Electronics and Communication Engineering?"},
    {"id": 22, "section": "B", "question": "Does AIT offer Mechanical and Robotics Engineering?"},
    {"id": 23, "section": "B", "question": "Does AIT offer Artificial Intelligence and Data Science?"},
    {"id": 24, "section": "B", "question": "What undergraduate programs are available apart from engineering?"},
    {"id": 25, "section": "B", "question": "Does AIT offer B.Sc Computer Science?"},

    # ── C. Postgraduate & Doctoral Programs (26–35) ──────────────────────────
    {"id": 26, "section": "C", "question": "What M.Tech programs are available?"},
    {"id": 27, "section": "C", "question": "What MBA specializations does AIT offer?"},
    {"id": 28, "section": "C", "question": "How long is the MBA program?"},
    {"id": 29, "section": "C", "question": "Does the MBA include an industry residency?"},
    {"id": 30, "section": "C", "question": "Does AIT offer an M.Tech in AI and Data Science?"},
    {"id": 31, "section": "C", "question": "Does AIT offer M.Tech in VLSI?"},
    {"id": 32, "section": "C", "question": "Does AIT offer M.Tech in AI and Software Architecture?"},
    {"id": 33, "section": "C", "question": "What PhD programs are available?"},
    {"id": 34, "section": "C", "question": "How often are PhD admissions conducted?"},
    {"id": 35, "section": "C", "question": "What entrance examination is required for PhD admission?"},

    # ── D. Academics & Curriculum (36–45) ────────────────────────────────────
    {"id": 36, "section": "D", "question": "What subjects are covered in B.Tech CSE?"},
    {"id": 37, "section": "D", "question": "What does the CSE curriculum look like in the first semester?"},
    {"id": 38, "section": "D", "question": "What programming languages are taught in CSE?"},
    {"id": 39, "section": "D", "question": "What subjects are taught in the third semester of CSE?"},
    {"id": 40, "section": "D", "question": "Does the CSE curriculum include Machine Learning?"},
    {"id": 41, "section": "D", "question": "Does CSE include Artificial Intelligence?"},
    {"id": 42, "section": "D", "question": "Does CSE include Cloud Computing?"},
    {"id": 43, "section": "D", "question": "What is the final-semester requirement for B.Tech CSE?"},
    {"id": 44, "section": "D", "question": "How are students evaluated?"},
    {"id": 45, "section": "D", "question": "What is the minimum CGPA required for graduation?"},

    # ── E. Admissions (46–55) ────────────────────────────────────────────────
    {"id": 46, "section": "E", "question": "What is the admission process at AIT?"},
    {"id": 47, "section": "E", "question": "What are the eligibility requirements for undergraduate admission?"},
    {"id": 48, "section": "E", "question": "What entrance exams does AIT accept?"},
    {"id": 49, "section": "E", "question": "What is AITSAT?"},
    {"id": 50, "section": "E", "question": "Is AITSAT required for every B.Tech applicant?"},
    {"id": 51, "section": "E", "question": "What is the AIT Research Entrance Test?"},
    {"id": 52, "section": "E", "question": "Can I apply for admission online?"},
    {"id": 53, "section": "E", "question": "What documents are required for admission?"},
    {"id": 54, "section": "E", "question": "Is there an application fee?"},
    {"id": 55, "section": "E", "question": "What is the last date to apply for admission?"},

    # ── F. Fees & Payments (56–65) ───────────────────────────────────────────
    {"id": 56, "section": "F", "question": "What is the B.Tech CSE tuition fee?"},
    {"id": 57, "section": "F", "question": "How much is the development fee?"},
    {"id": 58, "section": "F", "question": "What is the examination fee?"},
    {"id": 59, "section": "F", "question": "What are the hostel fees?"},
    {"id": 60, "section": "F", "question": "What is the cost of a single hostel room?"},
    {"id": 61, "section": "F", "question": "What is the cost of a shared hostel room?"},
    {"id": 62, "section": "F", "question": "What payment methods does AIT accept?"},
    {"id": 63, "section": "F", "question": "Can tuition fees be paid through UPI?"},
    {"id": 64, "section": "F", "question": "Can I pay fees through NEFT?"},
    {"id": 65, "section": "F", "question": "Can students pay their fees in installments?"},

    # ── G. Scholarships & Financial Aid (66–73) ──────────────────────────────
    {"id": 66, "section": "G", "question": "What scholarships are available at AIT?"},
    {"id": 67, "section": "G", "question": "Who is eligible for the Athena Merit Scholarship?"},
    {"id": 68, "section": "G", "question": "What percentage is required for the Athena Merit Scholarship?"},
    {"id": 69, "section": "G", "question": "What is the AITSAT rank requirement for the Merit Scholarship?"},
    {"id": 70, "section": "G", "question": "How much tuition waiver does the Merit Scholarship provide?"},
    {"id": 71, "section": "G", "question": "Is there a sports scholarship?"},
    {"id": 72, "section": "G", "question": "Who qualifies for the Sports Quota Scholarship?"},
    {"id": 73, "section": "G", "question": "Is there need-based financial aid?"},

    # ── H. Faculty & Administration (74–80) ──────────────────────────────────
    {"id": 74, "section": "H", "question": "Who is the Principal of AIT?"},
    {"id": 75, "section": "H", "question": "What is the Principal's message about academic excellence?"},
    {"id": 76, "section": "H", "question": "Who is the CSE HOD?"},
    {"id": 77, "section": "H", "question": "How can I contact the CSE HOD?"},
    {"id": 78, "section": "H", "question": "Who is the ECE HOD?"},
    {"id": 79, "section": "H", "question": "Who is the Mechanical and Robotics HOD?"},
    {"id": 80, "section": "H", "question": "Who is the MBA HOD?"},

    # ── I. Placements & Internships (81–90) ──────────────────────────────────
    {"id": 81, "section": "I", "question": "Does AIT have a placement cell?"},
    {"id": 82, "section": "I", "question": "What is AIT's placement percentage?"},
    {"id": 83, "section": "I", "question": "What is the average placement package?"},
    {"id": 84, "section": "I", "question": "What is the average CSE placement package?"},
    {"id": 85, "section": "I", "question": "What is the highest international package?"},
    {"id": 86, "section": "I", "question": "Which companies recruit from AIT?"},
    {"id": 87, "section": "I", "question": "Does Google recruit from AIT?"},
    {"id": 88, "section": "I", "question": "Does Microsoft recruit from AIT?"},
    {"id": 89, "section": "I", "question": "Does AIT provide internship opportunities?"},
    {"id": 90, "section": "I", "question": "What is the six-month industrial internship scheme?"},

    # ── J. Facilities (91–100) ───────────────────────────────────────────────
    {"id": 91, "section": "J", "question": "Does AIT have a central library?"},
    {"id": 92, "section": "J", "question": "What are the library timings?"},
    {"id": 93, "section": "J", "question": "Is the library open 24/7?"},
    {"id": 94, "section": "J", "question": "Does the library provide IEEE Xplore access?"},
    {"id": 95, "section": "J", "question": "Does the campus have Wi-Fi?"},
    {"id": 96, "section": "J", "question": "What sports facilities are available?"},
    {"id": 97, "section": "J", "question": "What are the gym timings?"},
    {"id": 98, "section": "J", "question": "Does the campus have a medical center?"},
    {"id": 99, "section": "J", "question": "What facilities are available in the NVIDIA DGX Suite?"},
    {"id": 100, "section": "J", "question": "What equipment is available in the VLSI and Embedded Systems Lab?"},
]


async def answer_v2_question(retriever, generator, question: str):
    """Query V2 retriever and generate grounded answer."""
    res = retriever.retrieve(query=question, candidate_k=15, final_k=3, min_threshold=0.20)
    answer = generator.generate(question=question, context=res.formatted_context, documents=res.selected_documents)
    sources = [
        f"{doc.title} [{doc.category.upper()}] (score={doc.rerank_score:.2f})"
        for doc in res.selected_documents
    ]
    return answer, sources, res


async def main():
    parser = argparse.ArgumentParser(description="Athena AI Chatbot — V2 Questions Runner")
    parser.add_argument("--ask", type=str, help="Ask an arbitrary single question directly to V2")
    parser.add_argument("--limit", "-l", type=int, default=0, help="Run only first N questions")
    parser.add_argument("--questions", "-q", type=str, help="Comma-separated IDs (e.g. 1,2,3)")
    parser.add_argument("--section", "-s", type=str, help="Run only specific section (A-J)")
    parser.add_argument("--delay", "-d", type=float, default=0.2, help="Delay between questions in seconds")
    parser.add_argument("--txt-output", default="test_results_v2.txt", help="TXT output file path")
    parser.add_argument("--json-output", default="test_results_v2.json", help="JSON output file path")
    args = parser.parse_args()

    retriever = get_retriever_v2()
    generator = get_grounded_generator_v2()

    # Interactive single query mode
    if args.ask:
        print(f"\n[V2 QUERY]: \"{args.ask}\"")
        print("-" * 70)
        answer, sources, res = await answer_v2_question(retriever, generator, args.ask)
        print(f"Sources ({len(sources)}):")
        for s in sources:
            print(f"  • {s}")
        print("\nAnswer:")
        print(answer)
        print("-" * 70 + "\n")
        return

    items = QUESTIONS

    # Filter by section
    if args.section:
        sec = args.section.strip().upper()
        items = [q for q in items if q["section"] == sec]

    # Filter by specific question IDs
    if args.questions:
        q_ids = {int(x.strip()) for x in args.questions.split(",") if x.strip().isdigit()}
        items = [q for q in items if q["id"] in q_ids]

    # Limit
    if args.limit and args.limit > 0:
        items = items[:args.limit]

    total = len(items)
    print("=" * 70)
    print(f"Athena AI Chatbot — V2 Architecture Runner ({total} Questions)")
    print(f"Target Collection: AthenaCollegeCollection_v2")
    print(f"Embedding Model:   text-embedding-3-large (3072 dims)")
    print(f"Started At:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"TXT Output :       {args.txt_output}")
    print(f"JSON Output:       {args.json_output}")
    print("=" * 70 + "\n")

    results = []
    txt_blocks = [
        f"ATHENA AI CHATBOT V2 — TEST RESULTS",
        f"Target Collection: AthenaCollegeCollection_v2",
        f"Embedding Model: text-embedding-3-large (3072 dims)",
        f"Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Questions: {total}",
        "=" * 80,
        ""
    ]

    for idx, item in enumerate(items, 1):
        q_id = item["id"]
        sec = item["section"]
        q_text = item["question"]

        print(f"[{idx}/{total}] Q#{q_id} (Sec {sec})")
        print(f"  Query  : {q_text}")

        try:
            start_time = time.perf_counter()
            answer, sources, res = await answer_v2_question(retriever, generator, q_text)
            elapsed = time.perf_counter() - start_time

            print(f"  Time   : {elapsed:.2f}s | Sources: {len(sources)}")
            for src in sources[:3]:
                print(f"    - {src}")
            print(f"  Answer : {answer[:140]}...\n")

            # Save result entry
            results.append({
                "id": q_id,
                "section": sec,
                "question": q_text,
                "answer": answer,
                "sources": sources,
                "duration_seconds": round(elapsed, 2),
            })

            # Append to text output
            txt_blocks.extend([
                f"Q#{q_id} (Section {sec}): {q_text}",
                "-" * 80,
                f"Answer:\n{answer}",
                "",
                f"Retrieved Sources ({len(sources)}):",
                "\n".join([f"  - {s}" for s in sources]) if sources else "  - None",
                "=" * 80,
                ""
            ])

            if args.delay > 0 and idx < total:
                await asyncio.sleep(args.delay)

        except Exception as err:
            print(f"  [ERROR]: {err}\n")
            results.append({
                "id": q_id,
                "section": sec,
                "question": q_text,
                "answer": f"Error: {err}",
                "sources": [],
                "duration_seconds": 0,
            })
            txt_blocks.extend([
                f"Q#{q_id} (Section {sec}): {q_text}",
                "-" * 80,
                f"ERROR: {err}",
                "=" * 80,
                ""
            ])

    # Write TXT file
    txt_path = Path(args.txt_output)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_blocks))

    # Write JSON file
    json_path = Path(args.json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print("V2 Execution complete!")
    print(f"Results saved to:")
    print(f"  - TXT  : {txt_path.resolve()}")
    print(f"  - JSON : {json_path.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
