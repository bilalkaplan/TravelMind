import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from cmu_rag_answer import get_llm_intent_and_location, generate_llm_answer, build_hotel_context, print_streamed_answer
from cmu_retrieve import search
from rich.console import Console

console = Console()

def run_test():
    query = "New York'ta tekerlekli sandalye erişimi olan, ücretsiz internet sunan lüks bir otel arıyorum. Bütçe önemli değil."
    print("Sorgu:", query)
    
    intent_data = get_llm_intent_and_location(query, [])
    intent = intent_data.get("intent", "hotel_search")
    location = intent_data.get("location", "New York")
    
    print(f"Intent: {intent}, Location: {location}")
    
    if intent == "hotel_search":
        print("Veritabanında aranıyor...")
        results = search(query)
        if not results:
            print("Otel bulunamadı.")
            return
            
        print(f"{len(results)} otel bulundu.")
        context_str = "\n".join([build_hotel_context(r, i+1) for i, r in enumerate(results[:5])])
        
        print("TravelMind Yanıtı oluşturuluyor...")
        answer = generate_llm_answer(query, context_str, [], location, "tr")
        for chunk in answer:
            print(chunk, end="", flush=True)
        print("\nTest bitti.")

if __name__ == '__main__':
    run_test()
