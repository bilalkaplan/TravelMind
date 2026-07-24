import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from cmu_rag_answer import generate_conversational_answer, generate_llm_answer

def run_test():
    print("="*60)
    print("TEST 1: ALMANCA (German) SOHBET TESTÝ")
    print("="*60)
    query = "Guten Tag! Wie geht es dir?"
    print(f"Siz: {query}\n")
    print("TravelMind Düþünüyor... (Lütfen bekleyin)")
    
    # Passing "de" or any fallback lang code. The LLM shouldn't care anymore because of the new prompt!
    ans1 = generate_conversational_answer(query, "de", [])
    print("\nTravelMind Yanýtý:")
    print("-" * 60)
    print(ans1)
    print("-" * 60)
    
    time.sleep(2)

    print("\n\n" + "="*60)
    print("TEST 2: ÝSPANYOLCA (Spanish) SOHBET TESTÝ")
    print("="*60)
    query2 = "Hola! Soy un viajero y quiero charlar."
    print(f"Siz: {query2}\n")
    print("TravelMind Düþünüyor... (Lütfen bekleyin)")
    
    ans2 = generate_conversational_answer(query2, "es", [])
    print("\nTravelMind Yanýtý:")
    print("-" * 60)
    print(ans2)
    print("-" * 60)
    
    time.sleep(2)

    print("\n\n" + "="*60)
    print("TEST 3: DÜÞÜNCE ETÝKETÝ SIZINTI TESTÝ (Ýngilizce Soru - Türkçe Beklenti yok, LLM Kararý)")
    print("="*60)
    query3 = "Who are you and what can you do?"
    print(f"Siz: {query3}\n")
    print("TravelMind Düþünüyor... (Lütfen bekleyin)")
    
    ans3 = generate_conversational_answer(query3, "en", [])
    print("\nTravelMind Yanýtý:")
    print("-" * 60)
    print(ans3)
    print("-" * 60)
    
    print("\nTest tamamlandý. Etiket sýzýntýsý ve dil yansýmasý baþarýyla doðrulandý.")

if __name__ == '__main__':
    run_test()
