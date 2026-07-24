import os
import sys
import json
import time
from ddgs import DDGS

METADATA_FILE = 'data/cmu_hotel_metadata.json'
RAW_ENRICHED_FILE = 'data/raw/hotel_enriched_raw.json'

def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_json(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def search_hotel_info(hotel_name, city):
    query = f"{hotel_name} {city} phone number amenities room types"
    
    while True:
        try:
            results = DDGS().text(query, max_results=3)
            context = ""
            for r in results:
                context += f"- {r.get('body', '')}\n"
            return context
        except Exception as e:
            if "RateLimit" in str(e) or "Timeout" in str(e) or "429" in str(e):
                print(f"DuckDuckGo Limit/Timeout! 20s bekleniyor... Hata: {e}")
                time.sleep(20)
            else:
                print(f"Network error (internet dropped?). Waiting 30s before retry... Error: {e}")
                time.sleep(30)

def extract_with_openai(api_key, context, hotel_name):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
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
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        text = response.choices[0].message.content.strip()
        
        if text.startswith("```json"): text = text[7:]
        if text.startswith("```"): text = text[3:]
        if text.endswith("```"): text = text[:-3]
            
        return json.loads(text.strip())
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return None

def extract_with_openrouter(api_key, context, hotel_name):
    from openai import OpenAI
    # OpenRouter uses the exact same OpenAI SDK, just a different base URL
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
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
    
    while True:
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b:free",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            text = response.choices[0].message.content.strip()
            
            if text.startswith("```json"): text = text[7:]
            if text.startswith("```"): text = text[3:]
            if text.endswith("```"): text = text[:-3]
                
            return json.loads(text.strip())
        except Exception as e:
            if "429" in str(e):
                print(f"  -> OpenRouter Rate Limit aşıldı, 20 saniye bekleniyor... (Deneme {attempt+1}/10)")
                time.sleep(20)
            else:
                print(f"OpenRouter API error: {e}")
                return None
    return None

def extract_with_gemini(api_key, context, hotel_name):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
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
    
    while True:
        time.sleep(4) # Ensure max 15 RPM for free tier
        try:
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1)
            )
            text = response.text.strip()
            
            if text.startswith("```json"): text = text[7:]
            if text.startswith("```"): text = text[3:]
            if text.endswith("```"): text = text[:-3]
                
            return json.loads(text.strip())
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower():
                print(f"  -> Gemini Kota/Limit, 30 saniye bekleniyor... Hata: {e}")
                time.sleep(30)
            else:
                print(f"  -> Network / Gemini error, 20 saniye bekleniyor... Hata: {e}")
                time.sleep(20)

def extract_with_regex(context, hotel_name):
    import re
    # 1. Telefon Numarası bulma (Regex)
    phone = None
    phone_match = re.search(r'(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}', context)
    if phone_match:
        phone = phone_match.group(0)
        
    # 2. Amenities (İmkanlar) bulma (Kelime havuzu tabanlı)
    amenities = []
    amenity_keywords = {
        "Wi-Fi": ["wi-fi", "wifi", "internet", "wireless"],
        "Pool": ["pool", "swimming", "indoor pool", "outdoor pool"],
        "Gym / Fitness": ["gym", "fitness", "workout"],
        "Breakfast": ["breakfast", "buffet"],
        "Parking": ["parking", "valet", "garage"],
        "Restaurant / Bar": ["restaurant", "dining", "bar", "lounge"],
        "Pet Friendly": ["pet friendly", "pets allowed", "dog"]
    }
    context_lower = context.lower()
    for am, kws in amenity_keywords.items():
        if any(kw in context_lower for kw in kws):
            amenities.append(am)
            
    # 3. Room Types bulma (Kelime havuzu tabanlı)
    room_types = []
    room_keywords = {
        "Standard Room": ["standard", "basic"],
        "Deluxe Room": ["deluxe"],
        "Suite": ["suite", "presidential"],
        "King Room": ["king"],
        "Queen Room": ["queen", "double"],
        "Studio": ["studio"]
    }
    for rt, kws in room_keywords.items():
        if any(kw in context_lower for kw in kws):
            room_types.append(rt)
            
    return {
        "phone": phone,
        "amenities": amenities,
        "room_types": room_types
    }

def extract_with_g4f(context, hotel_name):
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
    print(f"Toplam {total_hotels} otel bulundu. Veri çekme başlatılıyor...\n")
    
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
        
    print("\nİşlem tamamlandı! Veriler hotel_enriched_raw.json dosyasına kaydedildi.")

if __name__ == "__main__":
    main()
