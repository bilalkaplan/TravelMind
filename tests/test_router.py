import sys
import os
import pytest
from unittest.mock import patch, MagicMock

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.cmu_rag_answer import get_llm_intent_and_location

@patch('src.cmu_rag_answer.get_foundry_base_url', return_value="http://mock:1234/v1")
@patch('src.cmu_rag_answer.get_available_model_id', return_value="mock-model")
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
@patch('src.cmu_rag_answer.get_available_model_id', return_value="mock-model")
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
@patch('src.cmu_rag_answer.get_available_model_id', return_value="mock-model")
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
