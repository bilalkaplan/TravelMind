"""Run the required five foreground Detroit generation measurements."""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from answer_validator import validate_answer
from cmu_rag_answer import (
    build_hotel_context,
    generate_llm_answer,
)
from cmu_retrieve import search
from hotel_card_builder import build_hotel_cards


QUERY = "I'm looking for a hotel in Detroit, can you help me?"
LOCATION = "Detroit, MI"


def main() -> None:
    print("[BENCHMARK] Preparing Detroit retrieval context...", flush=True)
    results = search(QUERY, location_filter=LOCATION, top_k_hotels=12)
    cards = build_hotel_cards(
        results=results,
        user_query=QUERY,
        query_requirements={},
        requested_location=LOCATION,
    )
    cards = sorted(cards, key=lambda card: card.get("rank_score", 0.0), reverse=True)[:3]
    if not cards:
        raise RuntimeError("Detroit retrieval returned no hotel cards.")

    context = "\n\n".join(
        build_hotel_context(QUERY, card, index, lang_code="en")
        for index, card in enumerate(cards, start=1)
    )
    print(
        "[BENCHMARK] generator=grounded-card-summary "
        f"cards={[c['hotel_name'] for c in cards]}"
    )

    measurements = []
    for run_number in range(1, 6):
        started = time.perf_counter()
        first_token_seconds = None
        response = generate_llm_answer(
            query=QUERY,
            hotel_context_str=context,
            chat_history=[],
            location=LOCATION,
            lang_code="en",
            hotel_cards=cards,
        )
        answer_parts = []
        think_present = False
        for chunk in response:
            if isinstance(chunk, dict):
                think_present = think_present or chunk.get("type") == "think"
                content = chunk.get("content", "") if chunk.get("type") == "answer" else ""
            else:
                content = str(chunk)
            if not content:
                continue
            if first_token_seconds is None:
                first_token_seconds = time.perf_counter() - started
            answer_parts.append(content)
        total_seconds = time.perf_counter() - started

        clean_answer = "".join(answer_parts).strip()
        preamble_left_in_answer = bool(
            re.search(
                r"^\s*(?:okay\b|let me\b|i must\b|the user\b|"
                r"i know (?:the )?rules\b|<\/?(?:think|answer)\b)",
                clean_answer,
                re.I,
            )
        )
        preamble_removed = think_present

        validation = validate_answer(
            clean_answer,
            cards,
            "hotel_search",
            LOCATION,
            "en",
            allowed_hotel_names=[card.get("hotel_name") for card in cards],
        )
        measurement = {
            "run": run_number,
            "first_token_seconds": round(first_token_seconds or total_seconds, 6),
            "total_seconds": round(total_seconds, 6),
            "first_token_ms": round(
                (first_token_seconds or total_seconds) * 1000, 3
            ),
            "total_ms": round(total_seconds * 1000, 3),
            "answer_chars": len(clean_answer),
            "answer_words": len(clean_answer.split()),
            "preamble_removed": preamble_removed,
            "preamble_left_in_answer": preamble_left_in_answer,
            "fallback": validation["needs_fallback"],
            "issues": validation["issues"],
            "answer": validation["sanitized_answer"],
        }
        measurements.append(measurement)
        print("[BENCHMARK RUN] " + json.dumps(measurement, ensure_ascii=False), flush=True)

    print(
        "[BENCHMARK SUMMARY] "
        + json.dumps(
            {
                "runs": 5,
                "fallback_count": sum(item["fallback"] for item in measurements),
                "preamble_removed_count": sum(
                    item["preamble_removed"] for item in measurements
                ),
                "preamble_left_in_answer_count": sum(
                    item["preamble_left_in_answer"] for item in measurements
                ),
                "mean_first_token_seconds": round(
                    sum(item["first_token_seconds"] for item in measurements) / 5, 3
                ),
                "mean_total_seconds": round(
                    sum(item["total_seconds"] for item in measurements) / 5, 3
                ),
                "mean_first_token_ms": round(
                    sum(item["first_token_ms"] for item in measurements) / 5, 3
                ),
                "mean_total_ms": round(
                    sum(item["total_ms"] for item in measurements) / 5, 3
                ),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
