import json
import pandas as pd
import re

print("Loading data...")
with open('data/raw/hotel_enriched_raw.json', 'r', encoding='utf-8') as f:
    enriched_data = json.load(f)

reviews_df = pd.read_csv('data/processed/cmu_reviews_reliable.csv')
hotels_df = pd.read_csv('data/processed/cmu_hotels_reliable.csv')

def extract_with_regex(context):
    phone = None
    phone_match = re.search(r'(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}', context)
    if phone_match:
        phone = phone_match.group(0)
        
    amenities = []
    amenity_keywords = {
        "Wi-Fi": ["wi-fi", "wifi", "internet", "wireless"],
        "Pool": ["pool", "swimming", "indoor pool", "outdoor pool"],
        "Gym / Fitness": ["gym", "fitness", "workout"],
        "Breakfast": ["breakfast", "buffet"],
        "Parking": ["parking", "valet", "garage"],
        "Restaurant / Bar": ["restaurant", "dining", "bar", "lounge"],
        "Pet Friendly": ["pet friendly", "pets allowed", "dog", "dogs"]
    }
    context_lower = context.lower()
    for am, kws in amenity_keywords.items():
        if any(kw in context_lower for kw in kws):
            amenities.append(am)
            
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

missing_count = 0
fixed_count = 0

for hotel_key, data in enriched_data.items():
    if not data.get('amenities') and not data.get('room_types'):
        missing_count += 1
        hotel_name = data.get('hotel_name')
        
        # Find hotel ID
        hotel_match = hotels_df[hotels_df['hotel_name'] == hotel_name]
        if not hotel_match.empty:
            hotel_id = hotel_match.iloc[0]['hotel_id']
            hotel_reviews = reviews_df[reviews_df['hotel_id'] == hotel_id]
            
            # Combine all reviews into a single text block (up to 50000 chars for speed)
            combined_text = " ".join(hotel_reviews['review_text'].dropna().astype(str).tolist())[:50000]
            
            if combined_text:
                extracted = extract_with_regex(combined_text)
                if extracted['amenities'] or extracted['room_types']:
                    data['amenities'] = extracted['amenities']
                    data['room_types'] = extracted['room_types']
                    if extracted['phone']:
                        data['phone'] = extracted['phone']
                    fixed_count += 1

print(f"Total missing initially: {missing_count}")
print(f"Total fixed using reviews: {fixed_count}")

with open('data/raw/hotel_enriched_raw.json', 'w', encoding='utf-8') as f:
    json.dump(enriched_data, f, ensure_ascii=False, indent=2)

with open('data/cmu_hotel_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(enriched_data, f, ensure_ascii=False, indent=2)

print("Saved updated metadata to hotel_enriched_raw.json and cmu_hotel_metadata.json.")
