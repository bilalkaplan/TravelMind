"""Audit missing hotel metadata without modifying the dataset.

This file used to infer amenities and room types from guest-review prose and
then overwrite both ``hotel_enriched_raw.json`` and
``cmu_hotel_metadata.json``.  A review mentioning a pool, for example, is not
reliable evidence that the hotel currently offers one.  Keeping that behavior
would therefore corrupt the grounded facts used by the RAG application.

The script is retained only as a backwards-compatible, read-only audit tool.
It deliberately has no write mode.  Hotel metadata must be repaired from an
authoritative structured source through the data-enrichment pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA_PATH = PROJECT_ROOT / "data" / "raw" / "hotel_enriched_raw.json"


def _has_explicit_values(value: Any) -> bool:
    """Return whether a structured metadata field contains explicit values."""

    if isinstance(value, Mapping):
        return any(item not in (None, "", [], {}) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(item not in (None, "") for item in value)
    return value not in (None, "")


def audit_missing_metadata(metadata_path: Path) -> dict[str, Any]:
    """Inspect metadata and return missing-field counts without changing files."""

    with metadata_path.open("r", encoding="utf-8") as source:
        records = json.load(source)

    if not isinstance(records, dict):
        raise ValueError(f"Expected a JSON object in {metadata_path}")

    missing_amenities: list[str] = []
    missing_room_types: list[str] = []
    missing_both: list[str] = []

    for hotel_key, raw_record in records.items():
        record = raw_record if isinstance(raw_record, dict) else {}
        has_amenities = _has_explicit_values(record.get("amenities"))
        has_room_types = _has_explicit_values(record.get("room_types"))

        if not has_amenities:
            missing_amenities.append(str(hotel_key))
        if not has_room_types:
            missing_room_types.append(str(hotel_key))
        if not has_amenities and not has_room_types:
            missing_both.append(str(hotel_key))

    return {
        "source": str(metadata_path.resolve()),
        "hotel_count": len(records),
        "missing_amenities_count": len(missing_amenities),
        "missing_room_types_count": len(missing_room_types),
        "missing_both_count": len(missing_both),
        "missing_amenities": missing_amenities,
        "missing_room_types": missing_room_types,
        "missing_both": missing_both,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only audit for missing explicit hotel amenities and room types. "
            "This tool never infers facts from reviews and never writes metadata."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help=f"Metadata JSON to inspect (default: {DEFAULT_METADATA_PATH})",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Maximum missing hotel keys to show per category (default: 10)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sample_size = max(args.sample_size, 0)
    result = audit_missing_metadata(args.input)

    print("Hotel metadata audit (read-only)")
    print(f"Source: {result['source']}")
    print(f"Hotels: {result['hotel_count']}")
    print(f"Missing amenities: {result['missing_amenities_count']}")
    print(f"Missing room types: {result['missing_room_types_count']}")
    print(f"Missing both: {result['missing_both_count']}")

    if sample_size:
        for label, key in (
            ("Amenity sample", "missing_amenities"),
            ("Room-type sample", "missing_room_types"),
            ("Missing-both sample", "missing_both"),
        ):
            sample = result[key][:sample_size]
            if sample:
                print(f"{label}: {', '.join(sample)}")

    print(
        "No files were changed. Repair missing facts only from an authoritative "
        "structured source."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
