import sys
import os
import pytest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
# Add repo root so `import config` resolves the same way src/cmu_rag_answer.py expects it to
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
import src.cmu_rag_answer as rag_answer
from src.cmu_rag_answer import (
    build_amenity_followup_answer,
    build_grounded_followup_answer,
    fast_route_query,
    generate_conversational_answer,
    generate_followup_answer,
    get_available_model_id,
    get_llm_intent_and_location,
    next_hotel_index,
    resolve_hotel_selection,
)


ROUTER_CARDS = [
    {
        "hotel_name": "Alpha Hotel",
        "location": "San Jose, CA",
        "phone": "UNKNOWN",
        "hotel_class": "UNKNOWN",
        "travelmind_score": 81.2,
        "map_link": "UNKNOWN",
        "amenities": {"pool": "NO", "breakfast": "YES", "wifi": "YES"},
        "room_info": {"room_types": ["Queen Room"]},
    },
    {
        "hotel_name": "Arena Hotel",
        "location": "San Jose, CA",
        "phone": "+1 408 555 0100",
        "hotel_class": "3.0",
        "travelmind_score": 87.2,
        "map_link": "https://www.google.com/maps/search/?api=1&query=1,2",
        "amenities": {"pool": "YES", "breakfast": "YES", "wifi": "YES"},
        "room_info": {"room_types": ["King Room", "Suite"]},
    },
    {
        "hotel_name": "Cedar Inn",
        "location": "San Jose, CA",
        "phone": "UNKNOWN",
        "hotel_class": "2.0",
        "travelmind_score": 74.0,
        "map_link": "UNKNOWN",
        "amenities": {"pool": "YES", "breakfast": "UNKNOWN"},
        "room_info": {"room_types": []},
    },
]


@pytest.mark.parametrize(
    ("query", "expected_intent"),
    [
        ("hello", "general_chat"),
        ("hi how are you", "general_chat"),
        ("thanks", "general_chat"),
        ("show me the next option", "follow_up"),
        ("find me a flight to Boston", "out_of_scope"),
        ("what is the weather in Boston?", "out_of_scope"),
        ("how much is Arena Hotel per night?", "price_question"),
        ("how is the TravelMind score calculated?", "score_explanation"),
        ("what is the hotel class?", "class_explanation"),
        ("recommend a hotel", "missing_location"),
        ("find a hotel in Paris", "unsupported_location"),
        ("find a hotel in Boston", "hotel_search"),
        ("what do guests say about the rooms there?", "review_question"),
        ("any complaints about noise?", "review_question"),
        ("how is the service?", "review_question"),
        ("which one has a pool?", "followup_pool"),
        ("Does Arena Hotel have breakfast?", "followup_breakfast"),
        ("what is its phone number?", "specific_hotel_info"),
        ("tell me more", "specific_hotel_info"),
    ],
)
def test_fast_router_context_matrix(query, expected_intent):
    state = {"last_hotel_cards": ROUTER_CARDS, "selected_hotel_index": 1}
    assert fast_route_query(query, state)["intent"] == expected_intent


def test_initial_named_city_search_preserves_requested_hotel():
    route = fast_route_query(
        "Show Arena Hotel in San Jose",
        {"last_hotel_cards": ROUTER_CARDS, "selected_hotel_index": 0},
    )
    assert route["intent"] == "hotel_search"
    assert route["requested_hotel_name"] == "Arena Hotel"
    ui_source = open("ui/app.py", encoding="utf-8").read()
    assert 'requested_hotel_name=route_res.get("requested_hotel_name")' in ui_source


@pytest.mark.parametrize("selected", [-1, 99, "bad", None])
def test_invalid_selection_resets_to_first_card(selected):
    card, index = resolve_hotel_selection(ROUTER_CARDS, selected_index=selected)
    assert index == 0
    assert card["hotel_name"] == "Alpha Hotel"


def test_unnamed_and_named_selection_transitions():
    card, index = resolve_hotel_selection(
        ROUTER_CARDS, "what do guests say there?", selected_index=1
    )
    assert (card["hotel_name"], index) == ("Arena Hotel", 1)
    card, index = resolve_hotel_selection(
        ROUTER_CARDS, "how is Cedar Inn?", selected_index=1
    )
    assert (card["hotel_name"], index) == ("Cedar Inn", 2)
    assert next_hotel_index(ROUTER_CARDS, 0) == 1
    assert next_hotel_index(ROUTER_CARDS, 2) is None


def test_pool_across_results_names_only_confirmed_hotels():
    answer, index = build_amenity_followup_answer(
        ROUTER_CARDS, "pool", "which one has a pool?", selected_index=0
    )
    assert "Arena Hotel" in answer and "Cedar Inn" in answer
    assert "Alpha Hotel" not in answer
    assert index == 0


def test_named_breakfast_focuses_card_and_updates_selection():
    answer, index = build_amenity_followup_answer(
        ROUTER_CARDS, "breakfast", "Does Arena Hotel have breakfast?", selected_index=0
    )
    assert "Arena Hotel" in answer and "verified" in answer
    assert index == 1


@pytest.mark.parametrize(
    ("question", "selected", "expected", "index"),
    [
        ("What is Arena Hotel's phone number?", 0, "+1 408 555 0100", 1),
        ("What rooms does it have?", 1, "King Room", 1),
        ("What amenities does it have?", 1, "Wi-Fi", 1),
        ("What is Alpha Hotel's phone number?", 1, "does not include a verified phone", 0),
    ],
)
def test_grounded_followup_matrix(question, selected, expected, index):
    answer, resolved = build_grounded_followup_answer(
        question, ROUTER_CARDS, selected_index=selected
    )
    assert expected in answer
    assert resolved == index


def _completion_chunk(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content))]
    )


def test_conversational_greeting_skips_model(monkeypatch):
    monkeypatch.setattr(
        rag_answer,
        "create_chat_completion_with_retry",
        lambda request: pytest.fail("greeting must not start Foundry"),
    )
    chunks = list(generate_conversational_answer("hello", "en", []))
    assert chunks[0]["type"] == "answer"
    assert "TravelMind" in chunks[0]["content"]


@pytest.mark.parametrize(
    ("generator_name", "expected_max_tokens"),
    [("generate_conversational_answer", 120), ("generate_followup_answer", 180)],
)
def test_small_generators_disable_thinking_and_extract_answer(
    monkeypatch, generator_name, expected_max_tokens
):
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return [_completion_chunk("<answer>Grounded response")]

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(
        rag_answer,
        "create_chat_completion_with_retry",
        lambda request: request(client, "mock-model"),
    )
    if generator_name == "generate_conversational_answer":
        chunks = list(generate_conversational_answer("How can hotels help me?", "en", []))
    else:
        chunks = list(generate_followup_answer("Tell me more", "Hotel Card", "en", []))
    assert "".join(c["content"] for c in chunks if c["type"] == "answer") == "Grounded response"
    assert captured["max_tokens"] == expected_max_tokens
    assert captured["stop"] == ["</answer>"]
    assert captured["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert captured["messages"][-1]["content"].endswith("/no_think")


def test_followup_empty_stream_has_safe_nonempty_fallback(monkeypatch):
    monkeypatch.setattr(
        rag_answer, "create_chat_completion_with_retry", lambda request: []
    )
    chunks = list(generate_followup_answer("Tell me more", "Hotel Card", "en", []))
    answer = "".join(c["content"] for c in chunks if c["type"] == "answer")
    assert answer.strip()
    assert "reliable answer" in answer

# get_available_model_id now validates config.MODEL_ID against the endpoint
# instead of discovering/preferring among multiple cached models, so the
# mock returns the pinned model id rather than an arbitrary placeholder.
@patch('src.cmu_rag_answer.get_foundry_base_url', return_value="http://mock:1234/v1")
@patch('src.cmu_rag_answer.get_available_model_id', return_value=config.MODEL_ID)
@patch('src.cmu_rag_answer.OpenAI')
def test_router_clean_json(mock_openai, mock_get_model, mock_get_url):
    # Setup mock to return a clean JSON string
    mock_instance = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock(delta=MagicMock(content='{"intent": "hotel_search", "location": "Dallas, TX", "query_requirements": {"pool": "REQUIRED"}}'))]
    mock_instance.chat.completions.create.return_value = [mock_chunk]
    mock_openai.return_value = mock_instance

    res = get_llm_intent_and_location("dallas havuzlu otel", [])

    # Check if we got the fallback dict instead
    assert "pool" in res.get("query_requirements", {}), f"Failed, returned: {res}"
    assert res["intent"] == "hotel_search"
    assert res["location"] == "Dallas, TX"
    assert res["query_requirements"]["pool"] == "REQUIRED"

@patch('src.cmu_rag_answer.get_foundry_base_url', return_value="http://mock:1234/v1")
@patch('src.cmu_rag_answer.get_available_model_id', return_value=config.MODEL_ID)
@patch('src.cmu_rag_answer.OpenAI')
def test_router_dirty_json(mock_openai, mock_get_model, mock_get_url):
    mock_instance = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock(delta=MagicMock(content='Sure, here is your json:\n```json\n{"intent": "general_chat", "location": null}\n```'))]
    mock_instance.chat.completions.create.return_value = [mock_chunk]
    mock_openai.return_value = mock_instance

    res = get_llm_intent_and_location("merhaba", [])
    assert res["intent"] == "general_chat"

@patch('src.cmu_rag_answer.get_foundry_base_url', return_value="http://mock:1234/v1")
@patch('src.cmu_rag_answer.get_available_model_id', return_value=config.MODEL_ID)
@patch('src.cmu_rag_answer.OpenAI')
def test_router_fallback_heuristic(mock_openai, mock_get_model, mock_get_url):
    mock_instance = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock(delta=MagicMock(content='{"intent": "missing_location", "location": null}'))]
    mock_instance.chat.completions.create.return_value = [mock_chunk]
    mock_openai.return_value = mock_instance

    res = get_llm_intent_and_location("chicago'da kalacak otel", [])
    assert res["intent"] == "hotel_search"
    assert res["location"] == "Chicago, IL"


def test_get_available_model_id_returns_pinned_model_when_present():
    mock_client = MagicMock()
    mock_client.models.list.return_value = MagicMock(
        data=[MagicMock(id=config.MODEL_ID), MagicMock(id="some-other-model")]
    )
    assert get_available_model_id(mock_client) == config.MODEL_ID


def test_get_available_model_id_resolves_concrete_alias_variant():
    mock_client = MagicMock()
    concrete_id = "qwen3-4b-cuda-gpu:3"
    mock_client.models.list.return_value = MagicMock(
        data=[MagicMock(id=concrete_id)]
    )
    assert get_available_model_id(mock_client) == concrete_id


def test_get_available_model_id_raises_clear_error_when_missing():
    # Pinned model is NOT in the endpoint's model list -- must raise, not
    # silently substitute a different cached model.
    mock_client = MagicMock()
    mock_client.models.list.return_value = MagicMock(
        data=[MagicMock(id="qwen3-0.6b-generic-cpu:4")]
    )
    with pytest.raises(RuntimeError) as exc_info:
        get_available_model_id(mock_client)

    message = str(exc_info.value)
    assert config.MODEL_ID in message
    assert "scripts\\setup_foundry_runtime.py" in message
