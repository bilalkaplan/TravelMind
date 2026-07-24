with open('src/enrich_search_rag.py', 'r', encoding='utf-8') as f:
    code = f.read()

start_idx = code.find('def main():')
new_main = '''def extract_with_g4f(context, hotel_name):
    import g4f
    from g4f.client import Client
    import json
    import time
    
    # Instantiate without proxy, let it use whatever works
    client = Client()
    
    prompt = f"""
    You are a data extraction assistant. Based on the following web search snippets for the hotel "{hotel_name}", extract the following information and return it strictly as a JSON object.
    Do not return any markdown tags, just the raw JSON.
    If a piece of information is not found in the text, use null or an empty list.
    
    Expected JSON format:
    {{
      "phone": "String (the phone number)",
      "amenities": ["List", "of", "amenities"],
      "room_types": ["List", "of", "room", "types"]
    }}
    
    Web Search Snippets:
    {context}
    """
    
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.choices[0].message.content.strip()
            if text.startswith("```json"): text = text[7:]
            if text.startswith("```"): text = text[3:]
            if text.endswith("```"): text = text[:-3]
            return json.loads(text.strip())
        except Exception as e:
            print(f"  -> G4F LLM error, retrying in 3s... (Attempt {attempt+1}/5)")
            time.sleep(3)
    return None

def main():
    import sys
    import time
    import os
    import json
    
    print("TravelMind Veri Zenginleştirme Modülü")
    print("1) OpenRouter API (Tamamen Ücretsiz Modeller)")
    print("2) OpenAI (ChatGPT Plus) API")
    print("3) Google Gemini API")
    print("4) Offline NLP / Regex Extractor (API GEREKTİRMEZ - HIZLI)")
    print("5) G4F (GPT4Free) API - Banlanmayan Bedava LLM Ağı!")
    
    choice = "5"
    api_key = ""
    
    if len(sys.argv) >= 3:
        choice = sys.argv[1].strip()
        api_key = sys.argv[2].strip()
    elif len(sys.argv) == 2:
        choice = sys.argv[1].strip()
    else:
        choice = input("Lütfen kullanmak istediğiniz servisi seçin (1, 2, 3, 4 veya 5): ").strip()
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
        elif choice == "5":
            extracted_data = extract_with_g4f(context, hotel_name)
            time.sleep(2)
        
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
print('Fixed3')
