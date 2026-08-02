import sys
import os
import pytest

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.travelmind_scoring import calculate_travelmind_score

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

