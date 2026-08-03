import sys
import os
import json
import math
import pytest

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from scripts.build_review_stats import build_review_stats
from src.travelmind_scoring import (
    MAX_REVIEW_COUNT,
    WEIGHTS,
    calculate_travelmind_score,
    normalize_review_rating,
    score_review_volume,
)


EXPECTED_WEIGHTS = {
    "location_match": 15,
    "hotel_class": 10,
    "amenities_match": 15,
    "room_type_match": 12,
    "review_overall": 20,
    "review_service": 8,
    "review_rooms": 8,
    "review_cleanliness": 7,
    "review_volume": 5,
}


def test_weights_are_the_single_100_point_source_of_truth():
    assert WEIGHTS == EXPECTED_WEIGHTS
    assert sum(WEIGHTS.values()) == 100


def test_review_rating_and_volume_normalization():
    assert normalize_review_rating(1) == 0
    assert normalize_review_rating(3) == 50
    assert normalize_review_rating(5) == 100
    assert normalize_review_rating(None) is None

    score, _ = score_review_volume(99, maximum_review_count=999)
    assert score == pytest.approx(math.log1p(99) / math.log1p(999) * 100)


def test_missing_review_components_are_skipped_and_weight_is_renormalized():
    result = {"metadata": {"review_stats": {"overall": 3.0}}}

    scoring = calculate_travelmind_score("", result)

    assert scoring["travelmind_score"] == 50
    assert [component["key"] for component in scoring["components"]] == [
        "review_overall"
    ]


def test_numeric_review_scores_drive_the_weighted_score():
    result = {
        "metadata": {
            "location": "Detroit, MI",
            "hotel_class": "5",
            "amenities": ["Wi-Fi", "Pool", "Breakfast"],
            "room_types": ["Double Room"],
            "review_stats": {
                "overall": 5,
                "service": 4,
                "rooms": 3,
                "cleanliness": 2,
                "value": 1,
                "review_count": MAX_REVIEW_COUNT or 1,
            },
        }
    }

    scoring = calculate_travelmind_score(
        "Detroit hotel with pool and double room", result
    )

    assert scoring["travelmind_score"] == pytest.approx(88.75)
    scores_by_key = {
        component["key"]: component["score"]
        for component in scoring["components"]
    }
    assert scores_by_key["review_overall"] == 100
    assert scores_by_key["review_service"] == 75
    assert scores_by_key["review_rooms"] == 50
    assert scores_by_key["review_cleanliness"] == 25


def test_stats_builder_keeps_missing_ratings_as_none(tmp_path):
    raw_path = tmp_path / "review.txt"
    records = [
        {
            "offering_id": 42,
            "ratings": {"overall": 5, "service": 4, "value": 3},
        },
        {
            "offering_id": 42,
            "ratings": {"overall": 3, "service": 2},
        },
    ]
    raw_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    stats, parsed, skipped = build_review_stats(raw_path)

    assert parsed == 2
    assert skipped == 0
    assert stats["42"] == {
        "overall": 4.0,
        "service": 3.0,
        "rooms": None,
        "cleanliness": None,
        "value": 3.0,
        "review_count": 2,
    }

def test_calculate_travelmind_score():
    result = {
        "metadata": {
            "hotel_class": "4.5",
            "review_count_total": "2000"
        },
        "text": "This is a great hotel.",
        "score": 1.2
    }
    
    scoring = calculate_travelmind_score("", result)
    assert "travelmind_score" in scoring
    assert "components" in scoring
    assert len(scoring["components"]) > 0
    assert scoring["travelmind_score"] <= 100.0
    assert scoring["travelmind_score"] > 0.0

def test_calculate_travelmind_score_missing_meta():
    result = {
        "metadata": {},
        "text": "Okay hotel.",
        "score": 0.8
    }
    
    scoring = calculate_travelmind_score("", result)
    assert "travelmind_score" in scoring
    assert scoring["travelmind_score"] >= 0
