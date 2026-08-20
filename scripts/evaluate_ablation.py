"""Quantitative ablation check for the fact-gate validator.

Runs a fixed query set through the *real* runtime pipeline (search ->
build_hotel_cards -> build_hotel_context -> generate_llm_answer ->
validate_answer), then reports how often answer_validator.validate_answer()
found an issue in the model's raw output before any fallback was applied.

This intentionally reuses the validator's own structured `issues` list
(the same categories that gate real user-facing answers) rather than a
separate ad-hoc keyword scan, so the numbers describe what the live system
actually does instead of a simplified stand-in for it. Requires Foundry
Local to be running (`foundry service start`).
"""
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from cmu_retrieve import search, get_or_load_embedding_model, get_or_load_matrix, get_or_load_row_index
from cmu_rag_answer import build_hotel_context, generate_llm_answer
from hotel_card_builder import build_hotel_cards
from answer_validator import extract_final_answer, validate_answer

QUERIES = [
    ("Find me a nice hotel in Detroit with a pool.", "Detroit"),
    ("I need a hotel in Dallas.", "Dallas"),
    ("Looking for a cheap hotel in Seattle with breakfast.", "Seattle"),
    ("Any good places to stay in Boston?", "Boston"),
    ("Can you recommend a hotel in Miami with parking?", "Miami"),
    ("What's a good hotel in Chicago with wifi?", "Chicago"),
    ("I want a hotel in Denver with a gym.", "Denver"),
    ("Suggest a family-friendly hotel in Austin.", "Austin"),
    ("Looking for a business hotel in Houston.", "Houston"),
    ("Any hotels in Phoenix with free parking?", "Phoenix"),
]


def run_query(query, location):
    results = search(query, location_filter=location)
    if not results:
        return None

    cards = build_hotel_cards(
        results=results,
        user_query=query,
        query_requirements={},
        requested_location=location,
    )
    sorted_cards = sorted(cards, key=lambda c: c.get("rank_score", 0.0), reverse=True)[:3]
    if not sorted_cards:
        return None

    hotel_context_str = ""
    for i, card in enumerate(sorted_cards, start=1):
        hotel_context_str += build_hotel_context(query, card, i, lang_code="en") + "\n\n"

    gen = generate_llm_answer(query, hotel_context_str, [], location, "en", sorted_cards, {})
    full_answer = ""
    for chunk in gen:
        if isinstance(chunk, dict) and chunk.get("type") == "answer":
            full_answer += chunk["content"]

    raw_answer = extract_final_answer(full_answer)
    validation = validate_answer(raw_answer, sorted_cards, "hotel_search", location, "en")
    return raw_answer, validation


def main():
    print("Loading resources...")
    get_or_load_embedding_model()
    get_or_load_matrix()
    get_or_load_row_index()

    print("\nRunning Ablation Study: Fact-Gate (Validator) Impact")
    print("=" * 60)

    total = 0
    queries_with_any_issue = 0
    queries_with_blocking_issue = 0
    issue_type_counts = Counter()

    for query, location in QUERIES:
        print(f"\nQuery: '{query}'")
        outcome = run_query(query, location)
        if outcome is None:
            print("No results found; skipped.")
            continue

        raw_answer, validation = outcome
        total += 1
        issue_types = validation["issues"]
        blocking = validation["needs_fallback"]

        if issue_types:
            queries_with_any_issue += 1
        if blocking:
            queries_with_blocking_issue += 1
        for issue_type in issue_types:
            issue_type_counts[issue_type] += 1

        print(f"-> Raw output issues: {issue_types or 'none'}")
        print(f"-> Blocking (fallback triggered): {blocking}")

    print("\n" + "=" * 60)
    print("ABLATION REPORT SUMMARY")
    print("=" * 60)
    if total == 0:
        print("No queries produced results; nothing to report.")
        return

    print(f"Total queries evaluated: {total}")
    print(
        f"Raw model output contained a validator-flagged issue: "
        f"{queries_with_any_issue} / {total} ({(queries_with_any_issue / total) * 100:.1f}%)"
    )
    print(
        f"...of which the fact-gate replaced with a safe fallback (blocking): "
        f"{queries_with_blocking_issue} / {total} ({(queries_with_blocking_issue / total) * 100:.1f}%)"
    )
    if issue_type_counts:
        print("\nIssue type breakdown (raw output, before fact-gate):")
        for issue_type, count in issue_type_counts.most_common():
            print(f"  - {issue_type}: {count}")
    print("=" * 60)
    print(
        "Note: every answer that ultimately reaches the user has already had "
        "blocking issues replaced by validate_answer(); the numbers above "
        "describe what the raw, unvalidated model output looked like."
    )


if __name__ == "__main__":
    main()
