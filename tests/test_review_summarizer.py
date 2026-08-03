from src.review_summarizer import summarize_common_review_question


HOTEL = {
    "hotel_id": "90003",
    "hotel_name": "The Westin Detroit Metropolitan Airport",
}


def test_room_summary_uses_stats_and_only_observed_descriptors():
    chunks = [
        {"text": "The room was clean, comfortable, and extremely quiet."},
        {"text": "Our room was dark, although the bed was comfortable."},
    ]

    answer = summarize_common_review_question(
        HOTEL, chunks, "What do guests say about the rooms there?"
    )

    assert "4.64/5" in answer
    assert "clean rooms" in answer
    assert "comfortable rooms or beds" in answer
    assert "dark rooms" in answer
    assert len(answer.split(". ")) == 3


def test_noise_summary_does_not_turn_quiet_into_a_complaint():
    chunks = [
        {"text": "The rooms were extremely quiet despite being at the airport."},
        {"text": "We had no airport noise and slept well."},
    ]

    answer = summarize_common_review_question(
        HOTEL, chunks, "Any complaints about noise?"
    )

    assert "positive in the excerpts that address it" in answer
    assert "quiet rooms" in answer
    assert "does not prove that no complaints exist" in answer
    assert "Negative excerpts" not in answer


def test_service_summary_balances_majority_rating_and_specific_complaint():
    chunks = [
        {"text": "The staff were friendly and the service was excellent."},
        {"text": "We received good service from the front desk."},
        {"text": "POOR VALET service from start to finish; the valet was rude."},
    ]

    answer = summarize_common_review_question(
        HOTEL, chunks, "How is the service?"
    )

    assert "4.45/5" in answer
    assert "friendly staff" in answer
    assert "poor valet service" in answer
    assert "rude treatment" in answer


def test_positive_phrase_with_negation_is_not_reported_as_positive():
    chunks = [{"text": "The staff were not friendly and service was slow."}]

    answer = summarize_common_review_question(
        {"hotel_id": "missing", "hotel_name": "Example Hotel"},
        chunks,
        "How is the staff?",
    )

    assert "Positive excerpts specifically mention friendly staff" not in answer
    assert "slow service" in answer


def test_unrecognized_question_is_left_for_the_llm():
    assert summarize_common_review_question(
        HOTEL,
        [{"text": "A detailed review."}],
        "What surprised guests?",
    ) is None


def test_multi_aspect_question_answers_rooms_and_service():
    chunks = [
        {"text": "The room was clean and the staff were friendly."},
        {"text": "Service was excellent, although one room was dark."},
    ]

    answer = summarize_common_review_question(
        HOTEL, chunks, "How are the rooms and service?"
    )

    assert "room ratings average 4.64/5" in answer
    assert "service ratings average 4.45/5" in answer
    assert "do not treat top-k excerpts as a count" in answer


def test_room_service_is_not_misclassified_as_room_quality():
    answer = summarize_common_review_question(
        {"hotel_id": "missing", "hotel_name": "Example Hotel"},
        [{"text": "Room service was slow throughout the stay."}],
        "How was room service?",
    )

    assert "slow service" in answer
    assert "about the rooms" not in answer


def test_rating_without_matching_excerpt_does_not_infer_no_complaints():
    answer = summarize_common_review_question(
        HOTEL,
        [{"text": "The airport location was convenient."}],
        "How is the service?",
    )

    assert "4.45/5" in answer
    assert "without inferring particular strengths or complaints" in answer
