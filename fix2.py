with open('src/enrich_search_rag.py', 'r', encoding='utf-8') as f:
    code = f.read()

start_idx = code.find('def main():')
new_main = '''def main():
    import sys
    import time
    import os
    import json
    
    print("TravelMind Veri Zenginleştirme Modülü")
    print("1) OpenRouter API (Tamamen Ücretsiz Modeller)")
    print("2) OpenAI (ChatGPT Plus) API")
    print("3) Google Gemini API")
    print("4) Offline NLP / Regex Extractor (API GEREKTİRMEZ - HIZLI)")
    
    choice = "1"
    api_key = ""
    
    if len(sys.argv) >= 3:
        choice = sys.argv[1].strip()
        api_key = sys.argv[2].strip()
    elif len(sys.argv) == 2:
        choice = sys.argv[1].strip()
    else:
        choice = input("Lütfen kullanmak istediğiniz servisi seçin (1, 2, 3 veya 4): ").strip()
        if choice in ["1", "2", "3"]:
            api_key = input("Lütfen seçtiğiniz servisin API Key'ini yapıştırın: ").strip()
            if not api_key:
                print("API Key girmediniz, çıkış yapılıyor.")
                return
                
    METADATA_FILE = 'data/cmu_hotel_metadata.json'
    RAW_ENRICHED_FILE = 'data/raw/hotel_enriched_raw.json'
    
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
        
    total_hotels = len(metadata)
    print(f"Toplam {total_hotels} otel bulundu. Veri çekme başlatılıyor...\\n")
    
    if os.path.exists(RAW_ENRICHED_FILE):
        with open(RAW_ENRICHED_FILE, 'r', encoding='utf-8') as f:
            try:
                existing_data = json.load(f)
            except:
                existing_data = {}
    else:
        existing_data = {}
    
    i = 0
    for key, data in metadata.items():
        i += 1
        hotel_name = data.get("name")
        city = data.get("city")
        # Ensure we use the exact key format the JSON uses
        # Looking at original code: key = f"{hotel_name}::{city}" might be different than the dict key
        # Original code used: key in enriched_raw
        
        if key in existing_data:
            continue
            
        print(f"[{i}/{total_hotels}] Otel işleniyor: {hotel_name}")
        context = search_hotel_info(hotel_name, city)
        
        if not context:
            print(f"  -> {hotel_name} için arama sonucu bulunamadı.")
            continue
            
        extracted_data = None
        if choice == "1":
            extracted_data = extract_with_openrouter(api_key, context, hotel_name)
        elif choice == "2":
            extracted_data = extract_with_openai(api_key, context, hotel_name)
        elif choice == "3":
            extracted_data = extract_with_gemini(api_key, context, hotel_name)
        elif choice == "4":
            extracted_data = extract_with_regex(context, hotel_name)
            time.sleep(1)
        
        if not extracted_data:
            print(f"  -> {hotel_name} için veriyi çıkaramadı. Atlanıyor.")
            continue
            
        existing_data[key] = {
            "hotel_name": hotel_name,
            "city": city,
            "phone": extracted_data.get("phone"),
            "amenities": extracted_data.get("amenities", []),
            "room_types": extracted_data.get("room_types", [])
        }
        
        with open(RAW_ENRICHED_FILE, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
        
    print("\\nİşlem tamamlandı! Veriler hotel_enriched_raw.json dosyasına kaydedildi.")

if __name__ == "__main__":
    main()
'''

with open('src/enrich_search_rag.py', 'w', encoding='utf-8') as f:
    f.write(code[:start_idx] + new_main)
print('Fixed2')
