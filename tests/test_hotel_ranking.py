import pytest

import cmu_retrieve
from hotel_card_builder import build_hotel_cards


def _result(
    hotel_id,
    name,
    *,
    amenities=None,
    room_types=None,
    rating=3,
    location="Test City, TX",
):
    return {
        "chunk_id": f"test-{hotel_id}",
        "chunk_type": "hotel_profile",
        "text": "Static hotel profile evidence.",
        "vector_score": 0.5,
        "metadata": {
            "hotel_id": str(hotel_id),
            "hotel_name": name,
            "location": location,
            "hotel_class": "3",
            "amenities": [] if amenities is None else amenities,
            "room_types": [] if room_types is None else room_types,
            "review_stats": {
                "overall": rating,
                "service": rating,
                "rooms": rating,
                "cleanliness": rating,
                "value": rating,
                "review_count": 100,
            },
        },
    }


@pytest.fixture(autouse=True)
def _no_external_enrichment(monkeypatch):
    monkeypatch.setattr(cmu_retrieve, "get_full_hotel_metadata", lambda *args, **kwargs: {})


@pytest.mark.parametrize(
    ("feature", "matching_amenities", "missing_amenities", "matching_rooms", "missing_rooms"),
    [
        ("pool", ["Pool"], ["Gym"], [], []),
        ("breakfast", ["Breakfast"], ["Gym"], [], []),
        ("wifi", ["Wi-Fi"], ["Gym"], [], []),
        ("parking", ["Parking"], ["Gym"], [], []),
        ("single_room", [], [], ["Single Room"], ["King Room"]),
        ("double_room", [], [], ["King Room"], ["Single Room"]),
        ("suite", [], [], ["Suite"], ["King Room"]),
    ],
)
def test_required_feature_match_outranks_higher_review_score(
    feature,
    matching_amenities,
    missing_amenities,
    matching_rooms,
    missing_rooms,
):
    matching = _result(
        900001,
        "Matching Hotel",
        amenities=matching_amenities,
        room_types=matching_rooms,
        rating=1,
    )
    missing = _result(
        900002,
        "High Rated But Missing Hotel",
        amenities=missing_amenities,
        room_types=missing_rooms,
        rating=5,
    )

    cards = build_hotel_cards(
        [missing, matching],
        user_query=f"Test City hotel with {feature}",
        query_requirements={feature: "REQUIRED"},
        requested_location="Test City, TX",
    )

    assert cards[0]["hotel_name"] == "Matching Hotel"
    assert cards[0]["requirement_satisfaction"][feature] == "MATCH"
    assert cards[1]["requirement_satisfaction"][feature] == "MISSING"
    assert cards[0]["rank_score"] > cards[1]["rank_score"]


def test_empty_evidence_is_unknown_but_known_absence_is_missing():
    cards = build_hotel_cards(
        [
            _result(900003, "Unknown Hotel", amenities=[], room_types=[]),
            _result(900004, "Missing Hotel", amenities=["Gym"], room_types=["King Room"]),
        ],
        user_query="Test City hotel with pool and a suite",
        query_requirements={"pool": "REQUIRED", "suite": "REQUIRED"},
        requested_location="Test City, TX",
    )
    by_name = {card["hotel_name"]: card for card in cards}

    assert by_name["Unknown Hotel"]["requirement_satisfaction"] == {
        "pool": "UNKNOWN",
        "suite": "UNKNOWN",
    }
    assert by_name["Missing Hotel"]["requirement_satisfaction"] == {
        "pool": "MISSING",
        "suite": "MISSING",
    }
    assert by_name["Unknown Hotel"]["rank_score"] > by_name["Missing Hotel"]["rank_score"]


def test_optional_match_is_a_tiebreaker_and_none_has_no_effect():
    pool = _result(900005, "Pool Hotel", amenities=["Pool"])
    no_pool = _result(900006, "No Pool Hotel", amenities=["Gym"])

    optional_cards = build_hotel_cards(
        [no_pool, pool],
        user_query="Test City hotel; a pool would be nice",
        query_requirements={"pool": "OPTIONAL"},
        requested_location="Test City, TX",
    )
    assert optional_cards[0]["hotel_name"] == "Pool Hotel"
    assert optional_cards[0]["_optional_matches"] == 1
    assert optional_cards[1]["_rank_adjustment"] == 0

    none_cards = build_hotel_cards(
        [no_pool, pool],
        user_query="Test City hotel with pool",
        query_requirements={"pool": "NONE"},
        requested_location="Test City, TX",
    )
    assert all(
        card["requirement_satisfaction"]["pool"] == "NOT_REQUESTED"
        and card["_rank_adjustment"] == 0
        for card in none_cards
    )
    assert none_cards[0]["rank_score"] == pytest.approx(none_cards[1]["rank_score"])


def test_numeric_review_score_breaks_equal_requirement_ties():
    cards = build_hotel_cards(
        [
            _result(900007, "Low Review Hotel", amenities=["Wi-Fi"], rating=2),
            _result(900008, "High Review Hotel", amenities=["Wi-Fi"], rating=5),
        ],
        user_query="Test City hotel with wifi",
        query_requirements={"wifi": "REQUIRED"},
        requested_location="Test City, TX",
    )

    assert cards[0]["hotel_name"] == "High Review Hotel"
    assert cards[0]["travelmind_score"] > cards[1]["travelmind_score"]


def test_location_and_named_hotel_are_hard_card_constraints():
    cards = build_hotel_cards(
        [
            _result(900009, "Named Hotel", location="Other City, CA"),
            _result(900010, "Different Hotel", location="Test City, TX"),
            _result(900011, "Named Hotel", location="Test City, TX"),
        ],
        user_query="Tell me about Named Hotel in Test City",
        query_requirements={},
        requested_location="Test City, TX",
        requested_hotel_name="Named Hotel",
    )

    assert [card["hotel_id"] for card in cards] == ["900011"]


def test_keyword_overlap_score_rewards_exact_hotel_name_match():
    keyword_terms = {"marriott"}

    named_match = cmu_retrieve.keyword_overlap_score(
        keyword_terms, "Reviews mention a nice stay overall.", "Downtown Marriott"
    )
    no_match = cmu_retrieve.keyword_overlap_score(
        keyword_terms, "Reviews mention a nice stay overall.", "Downtown Hilton"
    )

    assert named_match > no_match
    assert named_match == pytest.approx(0.3)
    assert no_match == 0.0


def test_keyword_overlap_score_rewards_exact_text_match_over_none():
    keyword_terms = {"parking"}

    text_match = cmu_retrieve.keyword_overlap_score(
        keyword_terms, "Free parking is available on site.", "Some Hotel"
    )
    no_match = cmu_retrieve.keyword_overlap_score(
        keyword_terms, "Breakfast is included with every stay.", "Some Hotel"
    )

    assert text_match > 0.0
    assert no_match == 0.0


def test_keyword_overlap_score_is_zero_with_no_keyword_terms():
    assert cmu_retrieve.keyword_overlap_score(set(), "Free parking on site.", "Some Hotel") == 0.0
