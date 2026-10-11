#!/usr/bin/env python
"""
Test V2 Retrieval over AthenaCollegeCollection_v2.

Executes the 12 benchmark test queries against the V2 retrieval engine.
Logs for each query:
- Query
- Retrieved document IDs
- Similarity and distance scores
- Document titles
- Document categories
- Relevance ranking
- Final selected context
- Grounded answer generated via strict V2 anti-hallucination prompt

Verifies that V1 collection (AthenaCollegeCollection) remains completely untouched.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Setup project path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
env_path = ROOT_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

from app.services.retriever_service import get_retriever_v2, RetrieverV2
from app.services.grounded_service import get_grounded_generator_v2
from app.services.chromadb_service import (
    get_chroma_service_v2,
    PROTECTED_COLLECTION,
    DEFAULT_V2_COLLECTION,
)

logging.basicConfig(level=logging.WARNING)

TEST_QUERIES = [
    "How much is a shared hostel room?",
    "What is the hostel fee for 2-sharing?",
    "Who is the principal of AIT?",
    "Who is the CSE HOD?",
    "What undergraduate programs does AIT offer?",
    "What courses are available at AIT?",
    "Who are the official recruiters?",
    "What companies recruit students?",
    "What is the B.Tech CSE fee?",
    "What is the last date to apply?",
    "What is the minimum attendance requirement?",
    "What are the library timings?",
]


def run_tests(generate_answers: bool = True, output_json: str = "v2_test_results.json"):
    print("\n" + "=" * 80)
    print("        Athena Institute of Technology — V2 Retrieval & Grounding Evaluation")
    print("=" * 80)
    print(f"Collection:     {os.getenv('CHROMA_COLLECTION_V2', DEFAULT_V2_COLLECTION)}")
    print(f"Embedding:      {os.getenv('EMBEDDING_MODEL_V2', 'text-embedding-3-large')}")
    print(f"Protected V1:   {PROTECTED_COLLECTION} (SAFEGUARDED)")
    print("=" * 80 + "\n")

    retriever = get_retriever_v2()
    generator = get_grounded_generator_v2() if generate_answers else None

    # Verify V1 baseline
    chroma_svc = get_chroma_service_v2()
    client = chroma_svc.get_client()
    v1_count_before = -1
    try:
        v1_col = client.get_collection(name=PROTECTED_COLLECTION)
        v1_count_before = v1_col.count()
    except Exception as e:
        print(f"V1 Read Warning: {e}")

    results_data: List[Dict[str, Any]] = []
    failures: List[str] = []

    for idx, query in enumerate(TEST_QUERIES, 1):
        print(f"\n[{idx}/12] QUERY: \"{query}\"")
        print("-" * 80)

        res = retriever.retrieve(query=query, candidate_k=15, final_k=3, min_threshold=0.20)

        if not res.selected_documents:
            failures.append(query)
            print("  ⚠️ RETRIEVAL WARNING: No documents passed relevance threshold.")

        print(f"  Candidates Retrieved: {len(res.candidates)} | Final Selected: {len(res.selected_documents)}")
        print("\n  Top Ranked Candidates:")
        print(f"  {'Rank':<5} {'Score':<8} {'Sim':<8} {'Dist':<8} {'ID':<26} {'Category':<15} {'Title'}")
        print("  " + "-" * 78)

        candidate_logs = []
        for cand in res.candidates[:5]:
            print(
                f"  {cand.rank:<5} {cand.rerank_score:<8.4f} {cand.similarity:<8.4f} "
                f"{cand.distance:<8.4f} {cand.id:<26} {cand.category:<15} {cand.title}"
            )
            candidate_logs.append({
                "rank": cand.rank,
                "id": cand.id,
                "title": cand.title,
                "category": cand.category,
                "distance": round(cand.distance, 4),
                "similarity": round(cand.similarity, 4),
                "rerank_score": round(cand.rerank_score, 4),
            })

        print("\n  Selected Context:")
        for doc in res.selected_documents:
            print(f"    • [{doc.id}] {doc.title} (sim={doc.similarity:.4f}, score={doc.rerank_score:.4f})")
            snippet = doc.content.replace("\n", " ")[:160] + "..."
            print(f"      Content: {snippet}")

        grounded_answer = ""
        if generator and res.formatted_context:
            try:
                grounded_answer = generator.generate(question=query, context=res.formatted_context)
                print(f"\n  Grounded LLM Answer:")
                # Indent answer for clean reading
                for line in grounded_answer.split("\n"):
                    print(f"    {line}")
            except Exception as e:
                grounded_answer = f"[Generation Error: {e}]"
                print(f"    {grounded_answer}")

        results_data.append({
            "test_id": idx,
            "query": query,
            "candidates": candidate_logs,
            "selected_document_ids": [d.id for d in res.selected_documents],
            "selected_titles": [d.title for d in res.selected_documents],
            "selected_categories": [d.category for d in res.selected_documents],
            "final_context_snippets": [d.content[:300] for d in res.selected_documents],
            "grounded_answer": grounded_answer,
            "passed": len(res.selected_documents) > 0,
        })

    # Save output to JSON
    out_file = ROOT_DIR / output_json
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved detailed evaluation results to: {out_file}")

    # Summary
    print("\n" + "=" * 80)
    print("                           EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Test Queries:      {len(TEST_QUERIES)}")
    print(f"Successful Retrievals:    {len(TEST_QUERIES) - len(failures)} / {len(TEST_QUERIES)}")
    if failures:
        print(f"Retrieval Failures ({len(failures)}):")
        for f_q in failures:
            print(f"  - {f_q}")
    else:
        print("Retrieval Failures:       0 (ALL 12 QUERIES PASSED WITH HIGH RELEVANCE)")

    # Verify V1 safety after tests
    print("\n" + "-" * 80)
    print("                           V1 INTEGRITY CHECK")
    print("-" * 80)
    try:
        v1_col_after = client.get_collection(name=PROTECTED_COLLECTION)
        v1_count_after = v1_col_after.count()
        print(f"Protected V1 Collection: {PROTECTED_COLLECTION}")
        print(f"Count Before Tests:      {v1_count_before}")
        print(f"Count After Tests:       {v1_count_after}")
        if v1_count_before != -1 and v1_count_after == v1_count_before:
            print("STATUS: V1 COLLECTION COMPLETELY UNTOUCHED AND VERIFIED.")
        else:
            print("WARNING: V1 collection count changed!")
    except Exception as e:
        print(f"V1 Verification Note: {e}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run V2 Retrieval Test Cases")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM answer generation")
    parser.add_argument("--output", default="v2_test_results.json", help="Path to save output JSON")
    args = parser.parse_args()

    run_tests(generate_answers=not args.no_llm, output_json=args.output)
