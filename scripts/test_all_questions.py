#!/usr/bin/env python
"""
Automated Test Suite for Athena Institute of Technology AI Chatbot.
Evaluates all 100 benchmark questions across 10 categories (A through J).

Usage:
    # Run all 100 questions directly via LangGraph:
    python scripts/test_all_questions.py

    # Run a specific section (e.g., Section H - Faculty & Administration):
    python scripts/test_all_questions.py --section H

    # Run critical questions (e.g., #25, #55, #65, #74, #86, #87, #88):
    python scripts/test_all_questions.py --questions 25,55,65,74,86,87,88

    # Run first 5 questions for a quick check:
    python scripts/test_all_questions.py --limit 5

    # Run against a running HTTP server:
    python scripts/test_all_questions.py --http http://localhost:8000/api/chat

    # Custom report output:
    python scripts/test_all_questions.py --output reports/qa_benchmark.md
"""
import argparse
import asyncio
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time

# Ensure project root is on PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Ensure .env is explicitly loaded
from dotenv import load_dotenv
env_file = ROOT_DIR / ".env"
if env_file.exists():
    load_dotenv(dotenv_path=env_file)

# ── 100 Benchmark Questions Dataset ──────────────────────────────────────────

BENCHMARK_QUESTIONS = [
    # ── A. General Information & Campus Location (1–15) ──────────────────────
    {"id": 1, "section": "A", "category": "General & Location", "question": "Where is Athena Institute of Technology located?"},
    {"id": 2, "section": "A", "category": "General & Location", "question": "What are the standard administrative hours?"},
    {"id": 3, "section": "A", "category": "General & Location", "question": "What are the academic lecture hours?"},
    {"id": 4, "section": "A", "category": "General & Location", "question": "Is the college open on Sundays?"},
    {"id": 5, "section": "A", "category": "General & Location", "question": "What are the Saturday administrative hours?"},
    {"id": 6, "section": "A", "category": "General & Location", "question": "How can I reach the campus using public transportation?"},
    {"id": 7, "section": "A", "category": "General & Location", "question": "Which metro station is closest to the campus?"},
    {"id": 8, "section": "A", "category": "General & Location", "question": "Which bus routes stop at the college?"},
    {"id": 9, "section": "A", "category": "General & Location", "question": "Does the college provide a shuttle service?"},
    {"id": 10, "section": "A", "category": "General & Location", "question": "What time does the campus shuttle operate?"},
    {"id": 11, "section": "A", "category": "General & Location", "question": "What is the official address of AIT?"},
    {"id": 12, "section": "A", "category": "General & Location", "question": "What are the main campus facilities?"},
    {"id": 13, "section": "A", "category": "General & Location", "question": "Does AIT provide parking facilities?"},
    {"id": 14, "section": "A", "category": "General & Location", "question": "Is there an EV charging facility on campus?"},
    {"id": 15, "section": "A", "category": "General & Location", "question": "Where can students find bicycle parking?"},

    # ── B. Undergraduate Programs (16–25) ────────────────────────────────────
    {"id": 16, "section": "B", "category": "Undergraduate Programs", "question": "What B.Tech programs does AIT offer?"},
    {"id": 17, "section": "B", "category": "Undergraduate Programs", "question": "What undergraduate engineering programs are available?"},
    {"id": 18, "section": "B", "category": "Undergraduate Programs", "question": "What are all the B.Tech branches at AIT?"},
    {"id": 19, "section": "B", "category": "Undergraduate Programs", "question": "How long are the B.Tech programs?"},
    {"id": 20, "section": "B", "category": "Undergraduate Programs", "question": "Does AIT offer Computer Science Engineering?"},
    {"id": 21, "section": "B", "category": "Undergraduate Programs", "question": "Does AIT offer Electronics and Communication Engineering?"},
    {"id": 22, "section": "B", "category": "Undergraduate Programs", "question": "Does AIT offer Mechanical and Robotics Engineering?"},
    {"id": 23, "section": "B", "category": "Undergraduate Programs", "question": "Does AIT offer Artificial Intelligence and Data Science?"},
    {"id": 24, "section": "B", "category": "Undergraduate Programs", "question": "What undergraduate programs are available apart from engineering?"},
    {"id": 25, "section": "B", "category": "Undergraduate Programs", "question": "Does AIT offer B.Sc Computer Science?", "critical_note": "Should clearly state B.Sc is NOT offered (do not invent B.Sc program)."},

    # ── C. Postgraduate & Doctoral Programs (26–35) ──────────────────────────
    {"id": 26, "section": "C", "category": "Postgraduate & Doctoral", "question": "What M.Tech programs are available?"},
    {"id": 27, "section": "C", "category": "Postgraduate & Doctoral", "question": "What MBA specializations does AIT offer?"},
    {"id": 28, "section": "C", "category": "Postgraduate & Doctoral", "question": "How long is the MBA program?"},
    {"id": 29, "section": "C", "category": "Postgraduate & Doctoral", "question": "Does the MBA include an industry residency?"},
    {"id": 30, "section": "C", "category": "Postgraduate & Doctoral", "question": "Does AIT offer an M.Tech in AI and Data Science?"},
    {"id": 31, "section": "C", "category": "Postgraduate & Doctoral", "question": "Does AIT offer M.Tech in VLSI?"},
    {"id": 32, "section": "C", "category": "Postgraduate & Doctoral", "question": "Does AIT offer M.Tech in AI and Software Architecture?"},
    {"id": 33, "section": "C", "category": "Postgraduate & Doctoral", "question": "What PhD programs are available?"},
    {"id": 34, "section": "C", "category": "Postgraduate & Doctoral", "question": "How often are PhD admissions conducted?"},
    {"id": 35, "section": "C", "category": "Postgraduate & Doctoral", "question": "What entrance examination is required for PhD admission?"},

    # ── D. Academics & Curriculum (36–45) ────────────────────────────────────
    {"id": 36, "section": "D", "category": "Academics & Curriculum", "question": "What subjects are covered in B.Tech CSE?"},
    {"id": 37, "section": "D", "category": "Academics & Curriculum", "question": "What does the CSE curriculum look like in the first semester?"},
    {"id": 38, "section": "D", "category": "Academics & Curriculum", "question": "What programming languages are taught in CSE?"},
    {"id": 39, "section": "D", "category": "Academics & Curriculum", "question": "What subjects are taught in the third semester of CSE?"},
    {"id": 40, "section": "D", "category": "Academics & Curriculum", "question": "Does the CSE curriculum include Machine Learning?"},
    {"id": 41, "section": "D", "category": "Academics & Curriculum", "question": "Does CSE include Artificial Intelligence?"},
    {"id": 42, "section": "D", "category": "Academics & Curriculum", "question": "Does CSE include Cloud Computing?"},
    {"id": 43, "section": "D", "category": "Academics & Curriculum", "question": "What is the final-semester requirement for B.Tech CSE?"},
    {"id": 44, "section": "D", "category": "Academics & Curriculum", "question": "How are students evaluated?"},
    {"id": 45, "section": "D", "category": "Academics & Curriculum", "question": "What is the minimum CGPA required for graduation?"},

    # ── E. Admissions (46–55) ────────────────────────────────────────────────
    {"id": 46, "section": "E", "category": "Admissions", "question": "What is the admission process at AIT?"},
    {"id": 47, "section": "E", "category": "Admissions", "question": "What are the eligibility requirements for undergraduate admission?"},
    {"id": 48, "section": "E", "category": "Admissions", "question": "What entrance exams does AIT accept?"},
    {"id": 49, "section": "E", "category": "Admissions", "question": "What is AITSAT?"},
    {"id": 50, "section": "E", "category": "Admissions", "question": "Is AITSAT required for every B.Tech applicant?"},
    {"id": 51, "section": "E", "category": "Admissions", "question": "What is the AIT Research Entrance Test?"},
    {"id": 52, "section": "E", "category": "Admissions", "question": "Can I apply for admission online?"},
    {"id": 53, "section": "E", "category": "Admissions", "question": "What documents are required for admission?"},
    {"id": 54, "section": "E", "category": "Admissions", "question": "Is there an application fee?"},
    {"id": 55, "section": "E", "category": "Admissions", "question": "What is the last date to apply for admission?", "critical_note": "Should NOT fabricate a date; must advise contacting admissions."},

    # ── F. Fees & Payments (56–65) ───────────────────────────────────────────
    {"id": 56, "section": "F", "category": "Fees & Payments", "question": "What is the B.Tech CSE tuition fee?"},
    {"id": 57, "section": "F", "category": "Fees & Payments", "question": "How much is the development fee?"},
    {"id": 58, "section": "F", "category": "Fees & Payments", "question": "What is the examination fee?"},
    {"id": 59, "section": "F", "category": "Fees & Payments", "question": "What are the hostel fees?"},
    {"id": 60, "section": "F", "category": "Fees & Payments", "question": "What is the cost of a single hostel room?"},
    {"id": 61, "section": "F", "category": "Fees & Payments", "question": "What is the cost of a shared hostel room?"},
    {"id": 62, "section": "F", "category": "Fees & Payments", "question": "What payment methods does AIT accept?"},
    {"id": 63, "section": "F", "category": "Fees & Payments", "question": "Can tuition fees be paid through UPI?"},
    {"id": 64, "section": "F", "category": "Fees & Payments", "question": "Can I pay fees through NEFT?"},
    {"id": 65, "section": "F", "category": "Fees & Payments", "question": "Can students pay their fees in installments?", "critical_note": "Should clearly state available docs do not specify installment options."},

    # ── G. Scholarships & Financial Aid (66–73) ──────────────────────────────
    {"id": 66, "section": "G", "category": "Scholarships & Aid", "question": "What scholarships are available at AIT?"},
    {"id": 67, "section": "G", "category": "Scholarships & Aid", "question": "Who is eligible for the Athena Merit Scholarship?"},
    {"id": 68, "section": "G", "category": "Scholarships & Aid", "question": "What percentage is required for the Athena Merit Scholarship?"},
    {"id": 69, "section": "G", "category": "Scholarships & Aid", "question": "What is the AITSAT rank requirement for the Merit Scholarship?"},
    {"id": 70, "section": "G", "category": "Scholarships & Aid", "question": "How much tuition waiver does the Merit Scholarship provide?"},
    {"id": 71, "section": "G", "category": "Scholarships & Aid", "question": "Is there a sports scholarship?"},
    {"id": 72, "section": "G", "category": "Scholarships & Aid", "question": "Who qualifies for the Sports Quota Scholarship?"},
    {"id": 73, "section": "G", "category": "Scholarships & Aid", "question": "Is there need-based financial aid?"},

    # ── H. Faculty & Administration (74–80) ──────────────────────────────────
    {"id": 74, "section": "H", "category": "Faculty & Administration", "question": "Who is the Principal of AIT?", "critical_note": "Dr. Arthur Pendelton must be identified accurately."},
    {"id": 75, "section": "H", "category": "Faculty & Administration", "question": "What is the Principal's message about academic excellence?"},
    {"id": 76, "section": "H", "category": "Faculty & Administration", "question": "Who is the CSE HOD?"},
    {"id": 77, "section": "H", "category": "Faculty & Administration", "question": "How can I contact the CSE HOD?"},
    {"id": 78, "section": "H", "category": "Faculty & Administration", "question": "Who is the ECE HOD?"},
    {"id": 79, "section": "H", "category": "Faculty & Administration", "question": "Who is the Mechanical and Robotics HOD?"},
    {"id": 80, "section": "H", "category": "Faculty & Administration", "question": "Who is the MBA HOD?"},

    # ── I. Placements & Internships (81–90) ──────────────────────────────────
    {"id": 81, "section": "I", "category": "Placements & Internships", "question": "Does AIT have a placement cell?"},
    {"id": 82, "section": "I", "category": "Placements & Internships", "question": "What is AIT's placement percentage?"},
    {"id": 83, "section": "I", "category": "Placements & Internships", "question": "What is the average placement package?"},
    {"id": 84, "section": "I", "category": "Placements & Internships", "question": "What is the average CSE placement package?"},
    {"id": 85, "section": "I", "category": "Placements & Internships", "question": "What is the highest international package?"},
    {"id": 86, "section": "I", "category": "Placements & Internships", "question": "Which companies recruit from AIT?", "critical_note": "Must list documented recruiters (Google, Microsoft, Amazon, Meta, Apple, IBM, NVIDIA, Intel, etc.)."},
    {"id": 87, "section": "I", "category": "Placements & Internships", "question": "Does Google recruit from AIT?", "critical_note": "Must confirm Google is a documented recruiting partner."},
    {"id": 88, "section": "I", "category": "Placements & Internships", "question": "Does Microsoft recruit from AIT?", "critical_note": "Must confirm Microsoft is a documented recruiting partner."},
    {"id": 89, "section": "I", "category": "Placements & Internships", "question": "Does AIT provide internship opportunities?"},
    {"id": 90, "section": "I", "category": "Placements & Internships", "question": "What is the six-month industrial internship scheme?"},

    # ── J. Facilities (91–100) ───────────────────────────────────────────────
    {"id": 91, "section": "J", "category": "Campus Facilities", "question": "Does AIT have a central library?"},
    {"id": 92, "section": "J", "category": "Campus Facilities", "question": "What are the library timings?"},
    {"id": 93, "section": "J", "category": "Campus Facilities", "question": "Is the library open 24/7?"},
    {"id": 94, "section": "J", "category": "Campus Facilities", "question": "Does the library provide IEEE Xplore access?"},
    {"id": 95, "section": "J", "category": "Campus Facilities", "question": "Does the campus have Wi-Fi?"},
    {"id": 96, "section": "J", "category": "Campus Facilities", "question": "What sports facilities are available?"},
    {"id": 97, "section": "J", "category": "Campus Facilities", "question": "What are the gym timings?"},
    {"id": 98, "section": "J", "category": "Campus Facilities", "question": "Does the campus have a medical center?"},
    {"id": 99, "section": "J", "category": "Campus Facilities", "question": "What facilities are available in the NVIDIA DGX Suite?"},
    {"id": 100, "section": "J", "category": "Campus Facilities", "question": "What equipment is available in the VLSI and Embedded Systems Lab?"},
]


# ── Execution Logic ──────────────────────────────────────────────────────────

async def query_via_graph(question: str, session_id: str) -> dict:
    """Execute query directly through the Python LangGraph chatbot workflow."""
    from app.chatbot.graph import run_chat
    start = time.perf_counter()
    res = await run_chat(session_id=session_id, question=question)
    elapsed = time.perf_counter() - start
    return {
        "answer": res.get("answer", ""),
        "sources": res.get("sources", []),
        "duration": elapsed,
        "status": "200 OK",
    }


async def query_via_http(question: str, session_id: str, endpoint: str) -> dict:
    """Execute query via HTTP POST to the backend endpoint."""
    import httpx
    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(endpoint, json={"session_id": session_id, "message": question})
        elapsed = time.perf_counter() - start
        if resp.status_code == 200:
            data = resp.json()
            return {
                "answer": data.get("answer", ""),
                "sources": data.get("sources", []),
                "duration": elapsed,
                "status": f"{resp.status_code} OK",
            }
        else:
            return {
                "answer": f"Error: HTTP {resp.status_code} - {resp.text}",
                "sources": [],
                "duration": elapsed,
                "status": f"{resp.status_code} Error",
            }


# ── Main Runner ──────────────────────────────────────────────────────────────

async def run_benchmark(args: argparse.Namespace):
    questions = BENCHMARK_QUESTIONS

    # Filter by section
    if args.section:
        sec = args.section.upper()
        questions = [q for q in questions if q["section"] == sec]
        print(f"Filtered to Section {sec}: {len(questions)} questions")

    # Filter by question ID(s)
    if args.questions:
        q_ids = {int(x.strip()) for x in args.questions.split(",") if x.strip()}
        questions = [q for q in questions if q["id"] in q_ids]
        print(f"Filtered to Questions {q_ids}: {len(questions)} questions")

    # Limit
    if args.limit and args.limit > 0:
        questions = questions[:args.limit]
        print(f"Limited to first {args.limit} questions")

    total = len(questions)
    print("=" * 80)
    print(f" Athena AI Chatbot — Automated 100 Q&A Test Suite")
    print(f" Total questions to test: {total}")
    print(f" Execution Mode         : {'HTTP (' + args.http + ')' if args.http else 'Direct LangGraph'}")
    print(f" Timestamp              : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")

    results = []
    start_all = time.perf_counter()

    for idx, item in enumerate(questions, 1):
        q_id = item["id"]
        sec = item["section"]
        cat = item["category"]
        q_text = item["question"]
        note = item.get("critical_note")

        session_id = f"test_q_{q_id}_{int(time.time())}"
        print(f"[{idx}/{total}] Q#{q_id} (Sec {sec}: {cat})")
        print(f"  Query : {q_text}")
        if note:
            print(f"  Note  : {note}")

        try:
            if args.http:
                res = await query_via_http(q_text, session_id, args.http)
            else:
                res = await query_via_graph(q_text, session_id)

            duration = res["duration"]
            sources = res["sources"]
            answer = res["answer"]
            status = res["status"]

            print(f"  Status: {status} | Duration: {duration:.2f}s | Sources: {len(sources)}")
            for s in sources[:3]:
                src_title = s.get("source") or s.get("title") or "Doc"
                print(f"    - {src_title}")
            snippet = answer.replace("\n", " ")[:140]
            print(f"  Answer: {snippet}...\n")

            results.append({
                "id": q_id,
                "section": sec,
                "category": cat,
                "question": q_text,
                "critical_note": note,
                "status": status,
                "duration": duration,
                "sources_count": len(sources),
                "sources": [s.get("source") or s.get("title") or "Unknown" for s in sources],
                "answer": answer,
            })

            # Small pause between questions to avoid hitting rapid OpenAI bursts
            if args.delay > 0 and idx < total:
                await asyncio.sleep(args.delay)

        except Exception as exc:
            print(f"  FAILED: {exc}\n")
            results.append({
                "id": q_id,
                "section": sec,
                "category": cat,
                "question": q_text,
                "critical_note": note,
                "status": "Exception",
                "duration": 0,
                "sources_count": 0,
                "sources": [],
                "answer": f"Execution error: {exc}",
            })

    total_time = time.perf_counter() - start_all
    avg_time = total_time / total if total > 0 else 0
    passed_sources = sum(1 for r in results if r["sources_count"] > 0)

    print("=" * 80)
    print(" Execution Completed!")
    print(f" Total Questions Tested : {total}")
    print(f" Questions with Sources : {passed_sources} / {total} ({(passed_sources/total*100):.1f}%)")
    print(f" Total Time Elapsed     : {total_time:.1f}s (Average: {avg_time:.2f}s per query)")
    print("=" * 80)

    # ── Write Markdown Report ────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    md_lines = [
        "# Athena AI Chatbot — 100 Q&A Benchmark Report",
        "",
        f"- **Generated At:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Total Questions:** {total}",
        f"- **Questions with Retrieved Sources:** {passed_sources} / {total} ({(passed_sources/total*100 if total else 0):.1f}%)",
        f"- **Total Time:** {total_time:.1f}s (Avg {avg_time:.2f}s per question)",
        f"- **Mode:** {'HTTP (' + args.http + ')' if args.http else 'Direct LangGraph'}",
        "",
        "---",
        "",
    ]

    current_section = None
    for r in results:
        if r["section"] != current_section:
            current_section = r["section"]
            md_lines.append(f"## Section {current_section}: {r['category']}\n")

        note_str = f"\n> **Critical Requirement:** {r['critical_note']}\n" if r['critical_note'] else ""
        sources_md = ", ".join(f"`{s}`" for s in r['sources']) if r['sources'] else "_None (Direct answer)_"

        md_lines.extend([
            f"### #{r['id']}. {r['question']}",
            note_str,
            f"- **Status:** `{r['status']}` ({r['duration']:.2f}s)",
            f"- **Sources ({r['sources_count']}):** {sources_md}",
            "",
            "**Answer:**",
            r['answer'],
            "",
            "---",
            "",
        ])

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\nMarkdown report saved to: {out_path.resolve()}")

    # ── Optional JSON output ─────────────────────────────────────────────────
    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total": total,
                "questions_with_sources": passed_sources,
                "total_time_seconds": total_time,
                "results": results,
            }, f, indent=2, ensure_ascii=False)
        print(f"JSON data saved to: {json_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Athena AI Chatbot 100 Q&A Test Runner")
    parser.add_argument("--section", "-s", help="Run only specific section (A to J)")
    parser.add_argument("--questions", "-q", help="Run comma-separated question IDs (e.g., 25,55,65,74,86)")
    parser.add_argument("--limit", "-l", type=int, default=0, help="Limit to first N questions")
    parser.add_argument("--delay", "-d", type=float, default=0.2, help="Delay between questions in seconds (default: 0.2s)")
    parser.add_argument("--output", "-o", default="reports/qa_report.md", help="Path to markdown output report (default: reports/qa_report.md)")
    parser.add_argument("--json-output", "-j", default="reports/qa_results.json", help="Path to JSON output file (default: reports/qa_results.json)")
    parser.add_argument("--http", help="HTTP API endpoint URL (e.g. http://localhost:8000/api/chat). If omitted, runs direct Python LangGraph.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_benchmark(args))
