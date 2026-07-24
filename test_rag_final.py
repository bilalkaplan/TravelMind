import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from cmu_rag_answer import generate_conversational_answer

def run_test():
    print("="*80)
    print("TEST 1: ZENGİNLEŞTİRİLMİŞ HARİTA VE OLANAK (AMENITY) RAG TESTİ")
    print("="*80)
    query = "New York'ta tekerlekli sandalye erişimi olan, ücretsiz internet sunan lüks bir otel arıyorum. Bütçe önemli değil."
    print(f"Kullanıcı Sorusu: {query}\n")
    print("TravelMind RAG Motoru Çalışıyor... (Lütfen bekleyin)\n")
    
    # 2. parametre dil kodu, 3. parametre history
    answer = generate_conversational_answer(query, "tr", [])
    
    print("\n" + "="*80)
    print("TRAVELMIND YANITI:")
    print("="*80)
    for chunk in answer:
        print(chunk, end="", flush=True)
    print("\n" + "="*80)
    print("\nTest Tamamlandı!")

if __name__ == '__main__':
    run_test()
