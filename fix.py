with open('src/enrich_search_rag.py', 'r', encoding='utf-8') as f:
    code = f.read()

start_idx = code.find('def main():')
new_main = '''def main():
    import sys
    import pandas as pd
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
                
    df = pd.read_csv("data/raw/hotel_catalog.csv")
    total_hotels = len(df)
    
    print(f"Toplam {total_hotels} otel bulundu. Veri çekme başlatılıyor...\\n")
    
    output_file = "data/raw/hotel_enriched_raw.json"
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            try:
                existing_data = json.load(f)
            except:
                existing_data = {}
    else:
        existing_data = {}
    
    for i, row in df.iterrows():
        hotel_name = row['hotel_name']
        city = row['city']
        key = f"{hotel_name}::{city}"
        
        if key in existing_data:
            continue
            
        print(f"[{i+1}/{total_hotels}] Otel işleniyor: {hotel_name}")
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
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
        
    print("\\nİşlem tamamlandı! Veriler hotel_enriched_raw.json dosyasına kaydedildi.")

if __name__ == "__main__":
    main()
'''

with open('src/enrich_search_rag.py', 'w', encoding='utf-8') as f:
    f.write(code[:start_idx] + new_main)
print('Fixed')
