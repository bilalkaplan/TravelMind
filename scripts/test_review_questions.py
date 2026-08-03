"""Exercise the three required review-grounded questions end to end."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from answer_validator import extract_final_answer, validate_answer
from cmu_rag_answer import fast_route_query, generate_review_answer
from cmu_retrieve import search, search_reviews_for_hotel
from hotel_card_builder import build_hotel_cards


SEARCH_QUERY = "I'm looking for a hotel in Detroit, can you help me?"
LOCATION = "Detroit, MI"
QUESTIONS = [
    "what do guests say about the rooms there?",
    "any complaints about noise?",
    "how is the service?",
]
QUESTION_RATING_FIELD = {
    QUESTIONS[0]: "rooms",
    QUESTIONS[2]: "service",
}


def collect_answer(generator) -> str:
    parts = []
    for chunk in generator:
        if isinstance(chunk, dict) and chunk.get("type") == "answer":
            parts.append(str(chunk.get("content", "")))
        elif isinstance(chunk, str):
            parts.append(chunk)
    return extract_final_answer("".join(parts))


def main() -> None:
    with (PROJECT_ROOT / "data" / "hotel_review_stats.json").open(
        "r", encoding="utf-8"
    ) as stats_file:
        review_stats = json.load(stats_file)

    results = search(SEARCH_QUERY, location_filter=LOCATION, top_k_hotels=12)
    cards = build_hotel_cards(
        results=results,
        user_query=SEARCH_QUERY,
        query_requirements={},
        requested_location=LOCATION,
    )
    cards = sorted(cards, key=lambda card: card.get("rank_score", 0.0), reverse=True)[:3]
    if not cards:
        raise RuntimeError("Detroit retrieval returned no hotel cards.")

    session = {"last_hotel_cards": cards}
    fallback_count = 0
    for question in QUESTIONS:
        route = fast_route_query(question, session)
        if not route or route.get("intent") != "review_question":
            raise AssertionError(f"Review router failed for {question!r}: {route}")

        # The UI contract defaults to the top-ranked card when no hotel name
        # is explicitly present in the question.
        card = cards[0]
        chunks = search_reviews_for_hotel(card, question, k=8)
        if not chunks:
            raise AssertionError(f"No review evidence for {question!r}")

        answer = collect_answer(
            generate_review_answer(
                hotel_card=card,
                review_chunks=chunks,
                question=question,
                lang_code="en",
                chat_history=[],
            )
        )
        if not answer.strip():
            raise AssertionError(f"Review answer was empty for {question!r}")
        if "System Warning" in answer or "could not reach the local AI" in answer:
            raise RuntimeError(answer)
        if str(card.get("hotel_name")) not in answer:
            raise AssertionError("Answer did not identify the selected hotel.")
        sentence_count = len(
            [
                sentence
                for sentence in re.split(r"(?<=[.!?])\s+", answer)
                if sentence.strip()
            ]
        )
        if not 3 <= sentence_count <= 5:
            raise AssertionError(
                f"Expected 3-5 public sentences, got {sentence_count}: {answer}"
            )
        rating_field = QUESTION_RATING_FIELD.get(question)
        if rating_field:
            hotel_stats = review_stats.get(str(card.get("hotel_id")), {})
            expected_rating = hotel_stats.get(rating_field)
            if expected_rating is None or f"{float(expected_rating):.2f}/5" not in answer:
                raise AssertionError(
                    f"Answer did not use the stored {rating_field} mean."
                )
        elif "noise" not in answer.casefold():
            raise AssertionError("Noise answer did not address noise evidence.")

        validation = validate_answer(
            answer,
            [card],
            "review_question",
            None,
            "en",
            evidence_text=chunks,
            allowed_hotel_names=[card.get("hotel_name")],
        )
        fallback_count += int(validation["needs_fallback"])
        record = {
            "question": question,
            "intent": route["intent"],
            "hotel": card.get("hotel_name"),
            "review_chunks": len(chunks),
            "fallback": validation["needs_fallback"],
            "blocking": validation["blocking_issues"],
            "warnings": validation["warnings"],
            "answer": validation["sanitized_answer"],
        }
        print("[REVIEW TEST] " + json.dumps(record, ensure_ascii=False), flush=True)

    print(
        "[REVIEW SUMMARY] "
        + json.dumps({"questions": len(QUESTIONS), "fallback_count": fallback_count}),
        flush=True,
    )
    if fallback_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
