import sys
import os
import pytest

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.cmu_recommend_hotels import calculate_recommendation_score, extract_total_review_count

def test_extract_total_review_count():
    assert extract_total_review_count({}, "Total review count in CMU dataset: 500") == "500"
    assert extract_total_review_count({"review_count_total": "1200"}, "") == "1200"
    assert extract_total_review_count({"review_count_total": 450}, "") == 450
    assert extract_total_review_count({}, "No reviews here.") == ""

def test_calculate_recommendation_score():
    result = {
        "metadata": {
            "hotel_class": 4.5,
            "review_count_total": "2000"
        },
        "text": "This is a great hotel.",
        "score": 1.2
    }
    
    scoring = calculate_recommendation_score(result)
    assert "final_score" in scoring
    assert "class_points" in scoring
    assert scoring["class_points"] > 0
    assert scoring["review_points"] > 0
    assert scoring["final_score"] <= 10.0

def test_calculate_recommendation_score_missing_meta():
    result = {
        "metadata": {},
        "text": "Okay hotel.",
        "score": 0.8
    }
    
    scoring = calculate_recommendation_score(result)
    assert scoring["class_points"] == 0
    assert scoring["final_score"] > 0
