import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from cmu_rag_answer import (
    get_llm_intent_and_location, 
    generate_llm_answer, 
    generate_out_of_scope_answer, 
    generate_conversational_answer
)
from cmu_retrieve import search

def run_suite():
    with open("tests/test_suite.json", "r", encoding="utf-8") as f:
        suites = json.load(f)
        
    log_file = "ui_test_logs/test_results.md"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    total_q = sum(len(q) for q in suites.values())
    print(f"Running full test suite ({total_q} questions)...", flush=True)
    
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("# Otonom Arayüz Test Sonuçları (Backend)\n\n")
        
    for test_key, questions in suites.items():
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n## {test_key}\n\n")
            
        # 1 test key has 30 questions
        for i, q in enumerate(questions):
            print(f"[{test_key}] [{i+1}/{len(questions)}] {q}", flush=True)
            
            # Use 'tr' for test_1_tr to test_8_tr, and 'en' for test_9_en and test_10_en
            lang = "tr" if "_tr" in test_key else "en"
            
            # Step 1: Intent
            try:
                parsed = get_llm_intent_and_location(q, [])
                intent = parsed.get("intent")
                location = parsed.get("location")
                filters = parsed.get("filters", {})
                is_unsupported = (intent == "unsupported_location")
            except Exception as e:
                intent, location, is_unsupported, filters = "error", None, False, {}
                print("Error in intent:", e)
                
            if not isinstance(filters, dict):
                filters = {}
                
            # Step 2: Answer Generation
            full_answer = ""
            if intent == "out_of_scope":
                generator = generate_out_of_scope_answer(q, lang, [])
            elif is_unsupported:
                generator = generate_conversational_answer(q, lang, [])
            elif intent == "hotel_search":
                # Real search
                try:
                    search_results = search(q, location_filter=location, filters=filters, top_k_hotels=3)
                    generator = generate_llm_answer(q, search_results, [], location, lang)
                except Exception as e:
                    generator = [f"Search Error: {e}"]
            else:
                generator = generate_conversational_answer(q, lang, [])
                
            for chunk in generator:
                if isinstance(chunk, dict) and chunk.get("type") == "answer":
                    full_answer += chunk["content"]
                elif isinstance(chunk, str):
                    full_answer += chunk
                    
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"**Soru:** {q}\n")
                f.write(f"**Intent:** {intent} | **Location:** {location}\n")
                f.write(f"**Yanıt:**\n{full_answer.strip()}\n")
                f.write("---\n")

if __name__ == "__main__":
    run_suite()
