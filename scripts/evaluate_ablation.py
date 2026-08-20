import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from cmu_retrieve import search, get_or_load_embedding_model, get_or_load_matrix, get_or_load_row_index
from cmu_rag_answer import build_hotel_context, generate_llm_answer
from answer_validator import extract_final_answer, validate_answer

QUERIES = [
    ("Find me a nice hotel in Detroit with a pool.", "Detroit"),
    ("I need a hotel in Dallas.", "Dallas"),
    ("Looking for a cheap hotel in Seattle with breakfast.", "Seattle"),
    ("Any good places to stay in Boston?", "Boston"),
    ("Can you recommend a hotel in Miami with parking?", "Miami")
]

def main():
    print("Loading resources...")
    get_or_load_embedding_model()
    get_or_load_matrix()
    get_or_load_row_index()
    
    print("\nRunning Ablation Study: Fact-Gate (Validator) Impact")
    print("=" * 60)
    
    total = len(QUERIES)
    raw_leaks = 0
    validated_leaks = 0
    
    for query, location in QUERIES:
        print(f"\nQuery: '{query}'")
        results = search(query, location_filter=location)
        if not results:
            print("No results found.")
            continue
        
        cards = results[:3]
        hotel_context_str = ""
        for i, res in enumerate(cards, start=1):
            # Build card dict for testing
            card_dict = {
                "hotel_name": res["metadata"].get("hotel_name", "UNKNOWN"),
                "location": res["metadata"].get("location", "UNKNOWN"),
                "hotel_class": res["metadata"].get("hotel_class", ""),
                "phone": res["metadata"].get("phone", ""),
                "map_link": "",
                "score": res["score"],
                "amenities": res["metadata"].get("amenities", []),
                "room_types": res["metadata"].get("room_types", []),
                "missing_signals": ["parking"] if i == 2 else []
            }
            hotel_context_str += build_hotel_context(query, card_dict, i, lang_code="en") + "\n\n"
            
        gen = generate_llm_answer(query, hotel_context_str, [], location, "en", cards, {})
        full_answer = ""
        for chunk in gen:
            if isinstance(chunk, dict) and chunk.get("type") == "answer":
                full_answer += chunk["content"]
                
        raw_answer = extract_final_answer(full_answer)
        validation = validate_answer(raw_answer, cards, "hotel_search", location, "en")
        sanitized_answer = validation["sanitized_answer"]
        
        # Check leaks: dataset, database, our system, unknown names
        leak_keywords = ["dataset", "database", "I'm sorry", "unfortunately", "I am unable"]
        raw_has_leak = any(k.lower() in raw_answer.lower() for k in leak_keywords) or (raw_answer != sanitized_answer)
        val_has_leak = any(k.lower() in sanitized_answer.lower() for k in leak_keywords)
        
        if raw_has_leak:
            raw_leaks += 1
        if val_has_leak:
            validated_leaks += 1
            
        print(f"-> Raw Output had leaks/fallback triggered: {raw_has_leak}")
        print(f"-> Validated Output had leaks: {val_has_leak}")
        
    print("\n" + "=" * 60)
    print("ABLATION REPORT SUMMARY")
    print("=" * 60)
    print(f"Total queries evaluated: {total}")
    print(f"Errors WITHOUT Fact-Gate: {raw_leaks} / {total} ({(raw_leaks/total)*100:.1f}%)")
    print(f"Errors WITH Fact-Gate: {validated_leaks} / {total} ({(validated_leaks/total)*100:.1f}%)")
    print("=" * 60)

if __name__ == "__main__":
    main()
