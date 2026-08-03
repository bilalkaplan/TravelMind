"""Build per-hotel review rating statistics from the raw CMU dataset.

The input is large (roughly 1 GB), so records are aggregated in a single
streaming pass instead of being loaded into a dataframe.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "cmu_tripadvisor" / "review.txt"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "hotel_review_stats.json"
RATING_FIELDS = ("overall", "service", "rooms", "cleanliness", "value")


def _valid_rating(value: Any) -> float | None:
    """Return a finite rating on the CMU 1-5 scale, or ``None``."""

    if value is None or isinstance(value, bool):
        return None
    try:
        rating = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(rating) or not 1.0 <= rating <= 5.0:
        return None
    return rating


def build_review_stats(input_path: Path) -> tuple[dict[str, dict[str, float | int | None]], int, int]:
    """Aggregate review counts and available rating means by ``offering_id``.

    ``review_count`` counts every valid review record for a hotel. Each mean
    has its own denominator, so a missing sub-rating neither becomes zero nor
    changes the averages of the other fields.
    """

    aggregates: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "review_count": 0,
            "sums": {field: 0.0 for field in RATING_FIELDS},
            "counts": {field: 0 for field in RATING_FIELDS},
        }
    )
    parsed_records = 0
    skipped_records = 0

    with input_path.open("r", encoding="utf-8", errors="replace") as source:
        for line in source:
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                skipped_records += 1
                continue

            if not isinstance(record, dict):
                skipped_records += 1
                continue

            hotel_id = str(record.get("offering_id", "")).strip()
            if not hotel_id:
                skipped_records += 1
                continue

            parsed_records += 1
            aggregate = aggregates[hotel_id]
            aggregate["review_count"] += 1
            ratings = record.get("ratings")
            if not isinstance(ratings, dict):
                continue

            for field in RATING_FIELDS:
                rating = _valid_rating(ratings.get(field))
                if rating is None:
                    continue
                aggregate["sums"][field] += rating
                aggregate["counts"][field] += 1

    def hotel_sort_key(hotel_id: str) -> tuple[int, int | str]:
        return (0, int(hotel_id)) if hotel_id.isdigit() else (1, hotel_id)

    stats: dict[str, dict[str, float | int | None]] = {}
    for hotel_id in sorted(aggregates, key=hotel_sort_key):
        aggregate = aggregates[hotel_id]
        hotel_stats: dict[str, float | int | None] = {}
        for field in RATING_FIELDS:
            count = aggregate["counts"][field]
            hotel_stats[field] = (
                round(aggregate["sums"][field] / count, 4) if count else None
            )
        hotel_stats["review_count"] = aggregate["review_count"]
        stats[hotel_id] = hotel_stats

    return stats, parsed_records, skipped_records


def write_review_stats(
    stats: dict[str, dict[str, float | int | None]], output_path: Path
) -> None:
    """Write JSON atomically so an interrupted run cannot corrupt the output."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as target:
        json.dump(stats, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write("\n")
    temporary_path.replace(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-hotel CMU review averages and review counts."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Raw review file not found: {args.input}")

    stats, parsed_records, skipped_records = build_review_stats(args.input)
    write_review_stats(stats, args.output)
    max_review_count = max(
        (int(item["review_count"]) for item in stats.values()), default=0
    )

    print(f"Parsed reviews: {parsed_records}")
    print(f"Skipped records: {skipped_records}")
    print(f"Hotels: {len(stats)}")
    print(f"Maximum hotel review count: {max_review_count}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
