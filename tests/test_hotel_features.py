from pathlib import Path

import cmu_retrieve
import hotel_card_builder
from hotel_feature_verbalizer import (
    build_grounded_hotel_answer,
    coalesce_hotel_rewrite_sentences,
    get_recorded_room_types,
    get_verified_amenities,
    validate_hotel_feature_rewrite,
)


def test_arena_enriched_metadata_survives_empty_retrieval_fields():
    full = cmu_retrieve.get_full_hotel_metadata(
        "Arena Hotel", location="San Jose, CA"
    )
    merged = hotel_card_builder.merge_hotel_metadata(
        full,
        {
            "hotel_name": "Arena Hotel",
            "hotel_id": "119667",
            "location": "San Jose, CA",
            "amenities": [],
            "booking_room_types": [],
        },
    )

    assert merged["amenities"] == [
        "Wi-Fi",
        "Pool",
        "Gym / Fitness",
        "Breakfast",
        "Parking",
        "Restaurant / Bar",
    ]
    assert merged["room_types"] == [
        "Standard Room",
        "Suite",
        "King Room",
        "Queen Room",
    ]


def test_arena_card_exposes_all_verified_features_and_room_types():
    cards = hotel_card_builder.build_hotel_cards(
        results=[
            {
                "chunk_id": "arena-test",
                "chunk_type": "review",
                "text": "A historical review excerpt.",
                "vector_score": 0.9,
                "metadata": {
                    "hotel_name": "Arena Hotel",
                    "hotel_id": "119667",
                    "location": "San Jose, CA",
                    "amenities": [],
                    "booking_room_types": [],
                },
            }
        ],
        user_query="Arena Hotel with pool, breakfast and a king room",
        requested_location="San Jose, CA",
        query_requirements={
            "pool": "REQUIRED",
            "breakfast": "REQUIRED",
            "double_room": "REQUIRED",
        },
    )

    assert len(cards) == 1
    card = cards[0]
    assert get_verified_amenities(card, limit=None) == [
        "Wi-Fi",
        "pool",
        "breakfast",
        "parking",
        "gym/fitness facilities",
        "restaurant/bar",
    ]
    assert get_recorded_room_types(card, limit=None) == [
        "Standard Room",
        "Suite",
        "King Room",
        "Queen Room",
    ]
    assert card["requirement_satisfaction"] == {
        "pool": "MATCH",
        "breakfast": "MATCH",
        "double_room": "MATCH",
    }


def test_feature_verbalizer_uses_only_yes_and_explicit_room_type_lists():
    card = {
        "hotel_name": "Grounded Hotel",
        "travelmind_score": 80,
        "amenities": {
            "wifi": "YES",
            "pool": "NO",
            "breakfast": "UNKNOWN",
            "other": ["Restaurant / Bar"],
        },
        "room_info": {
            "suite": "YES",
            "room_types": [],
            "booking_room_types": ["Queen Room", "Queen Room"],
        },
    }

    assert get_verified_amenities(card) == ["Wi-Fi", "restaurant/bar"]
    assert get_recorded_room_types(card) == ["Queen Room"]
    answer = build_grounded_hotel_answer([card])
    assert "Wi-Fi and restaurant/bar" in answer
    assert "Queen Room" in answer
    assert "pool" not in answer.casefold()
    assert "breakfast" not in answer.casefold()
    assert "Suite" not in answer
    assert "available" not in answer.casefold()


def test_fact_gate_rejects_cross_hotel_feature_transfer():
    cards = [
        {
            "hotel_name": "Pool Hotel",
            "travelmind_score": 90,
            "amenities": {"pool": "YES"},
            "room_info": {"room_types": ["King Room"]},
        },
        {
            "hotel_name": "Breakfast Inn",
            "travelmind_score": 80,
            "amenities": {"breakfast": "YES"},
            "room_info": {"room_types": ["Queen Room"]},
        },
    ]
    canonical = build_grounded_hotel_answer(cards)
    assert validate_hotel_feature_rewrite(canonical, cards) == (True, [])

    transferred = canonical.replace(
        "verified amenities include breakfast",
        "verified amenities include breakfast and pool",
    )
    passed, reasons = validate_hotel_feature_rewrite(transferred, cards)
    assert passed is False
    assert any("unsupported amenity pool: Breakfast Inn" in reason for reason in reasons)


def test_fact_gate_preserves_one_sentence_per_displayed_hotel():
    cards = [
        {
            "hotel_name": "First Hotel",
            "travelmind_score": 90,
            "amenities": {"wifi": "YES"},
            "room_info": {"room_types": ["King Room"]},
        },
        {
            "hotel_name": "Second Inn",
            "travelmind_score": 80,
            "amenities": {"breakfast": "YES"},
            "room_info": {"room_types": ["Queen Room"]},
        },
    ]
    canonical = build_grounded_hotel_answer(cards)
    expanded = canonical.replace(
        "; verified amenities include Wi-Fi, while recorded room types",
        ". Verified amenities include Wi-Fi. Recorded room types",
    )
    passed, reasons = validate_hotel_feature_rewrite(expanded, cards)
    assert passed is False
    assert any("expected 2 sentences" in reason for reason in reasons)

    coalesced = coalesce_hotel_rewrite_sentences(expanded, cards)
    assert validate_hotel_feature_rewrite(coalesced, cards) == (True, [])


def test_ui_uses_static_record_wording_not_live_availability_claims():
    source = Path("ui/app.py").read_text(encoding="utf-8")
    assert "importlib.reload" not in source
    assert "from cmu_rag_answer import" in source
    assert "Verified amenities include" in source
    assert "Recorded room types include" in source
    assert "Exclusive privileges offered to our guests" not in source
    assert "similar room options are available" not in source
    assert "streamlit.components.v1" not in source
    assert "unsafe_allow_javascript=True" in source
