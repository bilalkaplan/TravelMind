import inspect

from src import cmu_rag_answer
from src import prompt_builders


class _Delta:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.delta = _Delta(content)


class _Chunk:
    def __init__(self, content):
        self.choices = [_Choice(content)]


def test_hotel_search_prompt_ends_with_exact_answer_contract():
    prompt = prompt_builders.build_final_answer_prompt(
        target_language="English",
        intent="hotel_search",
        requested_location="Detroit, MI",
        hotel_context_str="Hotel Card 1: Example Hotel",
        total_hotels_in_city=1,
        style_instruction=prompt_builders.get_style_instruction("English"),
    )

    assert prompt.rstrip().endswith(prompt_builders.ANSWER_WRAPPER_INSTRUCTION)
    assert "Assistant: <answer>Arena Hotel leads the recommendations" in prompt
    assert "verified amenities include breakfast and pool.</answer>" in prompt


def test_review_prompt_contains_only_evidence_contract_and_wrapper():
    prompt = prompt_builders.build_review_answer_prompt(
        {"hotel_name": "Example Hotel"},
        [
            {"text": "Rooms were spotless but street noise was noticeable."},
            {"chunk_text": "Service was friendly and quick."},
        ],
        "What do guests say about the rooms?",
    )

    assert "Rooms were spotless but street noise was noticeable." in prompt
    assert "Service was friendly and quick." in prompt
    assert "ONLY on the review evidence" in prompt
    assert "3 to 5 sentences" in prompt
    assert "both positive and negative" in prompt
    assert "booking or reservation" in prompt
    assert "<answer>your 3-to-5-sentence review answer here</answer>" in prompt


def test_review_prompt_compacts_long_excerpts_for_local_vram_budget():
    long_review = "Rooms were clean and quiet. " * 100
    prompt = prompt_builders.build_review_answer_prompt(
        {"hotel_name": "Example Hotel"},
        [{"text": long_review} for _ in range(8)],
        "What do guests say about the rooms?",
    )

    assert len(prompt) < 7_000
    assert prompt.count(" …") == 8


def test_extract_answer_accepts_closed_and_stop_truncated_wrappers():
    assert cmu_rag_answer.extract_answer(
        "Preamble<answer>The grounded answer.</answer>ignored"
    ) == "The grounded answer."
    assert cmu_rag_answer.extract_answer(
        "Preamble<answer>The grounded answer."
    ) == "The grounded answer."


def test_extract_answer_drops_only_a_meta_first_paragraph():
    raw = "Okay, let me inspect the cards.\n\nThe grounded answer."
    assert cmu_rag_answer.extract_answer(raw) == "The grounded answer."

    single_newline = "The user is asking about rooms.\nGuests liked the room size."
    assert cmu_rag_answer.extract_answer(single_newline) == "Guests liked the room size."

    normal = "Guests liked the rooms.\n\nSome mentioned noise."
    assert cmu_rag_answer.extract_answer(normal) == normal


def test_extract_answer_drops_repeated_leading_meta_paragraphs():
    raw = (
        "I must check the rules first.\n\n"
        "The user is asking for a hotel.\n\n"
        "I found three grounded hotel options."
    )
    assert cmu_rag_answer.extract_answer(raw) == (
        "I found three grounded hotel options."
    )


def test_structured_hotel_cards_use_non_thinking_qwen_and_fact_gate(monkeypatch):
    captured = {}
    cards = [
        {
            "hotel_name": "Arena Hotel",
            "location": "San Jose, CA",
            "travelmind_score": 87.25,
            "amenities": {
                "wifi": "YES",
                "pool": "YES",
                "breakfast": "YES",
                "parking": "YES",
                "other": ["Gym / Fitness", "Restaurant / Bar"],
            },
            "room_info": {
                "room_types": [
                    "Standard Room", "Suite", "King Room", "Queen Room"
                ]
            },
        },
        {
            "hotel_name": "Second Inn",
            "travelmind_score": 80.0,
            "amenities": {"wifi": "YES"},
            "room_info": {"room_types": ["Queen Room"]},
        },
        {
            "hotel_name": "Third Suites",
            "travelmind_score": 75.0,
            "amenities": {"breakfast": "YES"},
            "room_info": {"room_types": ["Suite"]},
        },
    ]
    canonical = cmu_rag_answer.safe_card_based_fallback_answer(
        hotel_cards=cards, city="San Jose, CA"
    )

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return [_Chunk("<answer>" + canonical)]

    class _Client:
        class _Chat:
            completions = _Completions()

        chat = _Chat()

    monkeypatch.setattr(
        cmu_rag_answer,
        "create_chat_completion_with_retry",
        lambda make_request: make_request(_Client(), "test-model"),
    )

    chunks = list(
        cmu_rag_answer.generate_llm_answer(
            "Hotels in Detroit",
            "legacy context",
            [{"role": "user", "content": "SECRET HISTORY"}],
            "San Jose, CA",
            hotel_cards=cards,
        )
    )

    assert chunks == [{"type": "answer", "content": canonical}]
    assert captured["stop"] == ["</answer>"]
    assert captured["temperature"] == 0.3
    assert captured["max_tokens"] == 320
    assert captured["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    factual_draft = captured["messages"][1]["content"]
    assert "Wi-Fi, pool, breakfast, parking" in factual_draft
    assert "Standard Room, Suite, King Room, and Queen Room" in factual_draft
    assert "SECRET HISTORY" not in str(captured["messages"])


def test_structured_hotel_rewrite_with_invented_spa_uses_grounded_fallback(
    monkeypatch,
):
    cards = [{
        "hotel_name": "Arena Hotel",
        "location": "San Jose, CA",
        "travelmind_score": 87.25,
        "amenities": {"wifi": "YES", "pool": "YES"},
        "room_info": {"room_types": ["King Room"]},
    }]
    canonical = cmu_rag_answer.safe_card_based_fallback_answer(
        hotel_cards=cards, city="San Jose, CA"
    )

    class _Completions:
        def create(self, **_kwargs):
            invented = canonical.replace("pool", "pool and spa")
            return [_Chunk("<answer>" + invented)]

    class _Client:
        class _Chat:
            completions = _Completions()

        chat = _Chat()

    monkeypatch.setattr(
        cmu_rag_answer,
        "create_chat_completion_with_retry",
        lambda make_request: make_request(_Client(), "test-model"),
    )

    chunks = list(
        cmu_rag_answer.generate_llm_answer(
            "Hotels in San Jose",
            "legacy context",
            [],
            "San Jose, CA",
            hotel_cards=cards,
        )
    )
    assert chunks == [{"type": "answer", "content": canonical}]
    assert "spa" not in chunks[0]["content"].casefold()


def test_structured_hotel_summary_flags_unmet_requested_features():
    cards = [{
        "hotel_name": "First Hotel",
        "travelmind_score": 80.0,
        "requirement_satisfaction": {
            "breakfast": "MISSING",
            "wifi": "UNKNOWN",
        },
    }]

    answer = cmu_rag_answer.safe_card_based_fallback_answer(
        user_query="Hotel with breakfast and Wi-Fi",
        hotel_cards=cards,
        city="Detroit",
        query_requirements={"breakfast": "REQUIRED", "wifi": "REQUIRED"},
    )
    assert "requested breakfast is not listed in its profile" in answer
    assert "requested Wi-Fi could not be confirmed from its profile" in answer


def test_hotel_context_contains_verified_amenities_and_static_room_types():
    context = cmu_rag_answer.build_hotel_context(
        "Hotels in San Jose",
        {
            "hotel_name": "Arena Hotel",
            "location": "San Jose, CA",
            "hotel_class": 3.0,
            "travelmind_score": 87.25,
            "amenities": {
                "wifi": "YES",
                "pool": "YES",
                "breakfast": "YES",
                "parking": "YES",
                "other": ["Gym / Fitness", "Restaurant / Bar"],
            },
            "room_info": {
                "room_types": [
                    "Standard Room", "Suite", "King Room", "Queen Room"
                ]
            },
        },
        1,
    )

    assert "Verified amenities: Wi-Fi, pool, breakfast, parking" in context
    assert "gym/fitness facilities" in context
    assert "restaurant/bar" in context
    assert "Standard Room, Suite, King Room, and Queen Room" in context
    assert "not live availability" in context


def test_stream_extract_answer_buffers_and_removes_opening_tag():
    response = [
        _Chunk("<thi"),
        _Chunk("nk>private reasoning</think>Okay, I must answer.\n\n<ans"),
        _Chunk("wer>Guests generally found the rooms clean."),
    ]

    chunks = list(cmu_rag_answer.stream_extract_answer(response, "en"))
    assert "".join(c["content"] for c in chunks if c["type"] == "think") == "private reasoning"
    assert [c for c in chunks if c["type"] == "answer"] == [
        {"type": "answer", "content": "Guests generally found the rooms clean."}
    ]


def test_generate_review_answer_uses_stop_and_extraction(monkeypatch):
    captured = {}

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return [_Chunk("<answer>Guests praised the service.")]

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    def run_request(make_request):
        return make_request(_Client(), "test-model")

    monkeypatch.setattr(
        cmu_rag_answer,
        "create_chat_completion_with_retry",
        run_request,
    )

    result = list(
        cmu_rag_answer.generate_review_answer(
            {"hotel_name": "Example Hotel"},
            [{"text": "The service was excellent."}],
            "What surprised guests?",
        )
    )

    assert captured["stop"] == ["</answer>"]
    assert captured["max_tokens"] == 180
    assert captured["messages"][-1]["content"].endswith("/no_think")
    assert captured["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert result == [
        {"type": "answer", "content": "Guests praised the service."}
    ]


def test_review_generator_never_returns_an_empty_public_answer(monkeypatch):
    class _Completions:
        def create(self, **_kwargs):
            return [_Chunk("<think>unfinished private reasoning")]

    class _Client:
        class _Chat:
            completions = _Completions()

        chat = _Chat()

    monkeypatch.setattr(
        cmu_rag_answer,
        "create_chat_completion_with_retry",
        lambda make_request: make_request(_Client(), "test-model"),
    )

    chunks = list(
        cmu_rag_answer.generate_review_answer(
            {"hotel_name": "Example Hotel"},
            [{"text": "The room was clean."}],
            "What surprised guests?",
        )
    )

    public_answer = "".join(
        chunk["content"] for chunk in chunks if chunk["type"] == "answer"
    )
    assert "did not produce a reliable answer" in public_answer


def test_all_answer_generators_define_the_closing_tag_stop_sequence():
    for generator in (
        cmu_rag_answer.generate_llm_answer,
        cmu_rag_answer.generate_followup_answer,
        cmu_rag_answer.generate_conversational_answer,
        cmu_rag_answer.generate_review_answer,
    ):
        assert 'stop=["</answer>"]' in inspect.getsource(generator)


def test_fast_router_catches_required_review_questions():
    state = {"last_hotel_cards": [{"hotel_name": "The Example Hotel"}]}

    for question in (
        "what do guests say about the rooms there?",
        "any complaints about noise?",
        "how is the service?",
    ):
        assert cmu_rag_answer.fast_route_query(question, state)["intent"] == "review_question"

    named = cmu_rag_answer.fast_route_query(
        "What do guests say about Example Hotel?",
        state,
    )
    assert named == {
        "intent": "review_question",
        "requested_hotel_name": "The Example Hotel",
    }
