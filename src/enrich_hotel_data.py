import json
import time
import requests
from pathlib import Path

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_PATH = PROCESSED_DIR / "cmu_chunks.jsonl"
OUTPUT_PATH = DATA_DIR / "cmu_hotel_metadata.json"

def get_unique_hotels():
    hotels = {}
    if not CHUNKS_PATH.exists():
        print(f"Error: {CHUNKS_PATH} not found.")
        return hotels
        
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            meta = item.get("metadata", {})
            name = meta.get("hotel_name")
            city = meta.get("location")
            if name and city:
                key = f"{name}::{city}"
                if key not in hotels:
                    hotels[key] = {"name": name, "city": city}
    return list(hotels.values())

def fetch_metadata_nominatim(hotel_name, city):
    """
    Fetches basic metadata (coordinates, exact address) using OpenStreetMap Nominatim API.
    Nominatim is free and requires NO API KEY, but STRICTLY requires 1 request per second.
    """
    url = "https://nominatim.openstreetmap.org/search"
    headers = {
        "User-Agent": "TravelMind-RAG-Internship-Project/1.0 (contact: test@example.com)"
    }
    params = {
        "q": f"{hotel_name}, {city}, USA",
        "format": "json",
        "addressdetails": 1,
        "limit": 1
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if len(data) > 0:
                result = data[0]
                return {
                    "lat": result.get("lat"),
                    "lon": result.get("lon"),
                    "address": result.get("display_name"),
                    "category": result.get("category"),
                    "type": result.get("type")
                }
    except Exception as e:
        print(f"Error fetching {hotel_name}: {e}")
        
    return None

def main():
    print("Loading unique hotels from chunks...")
    hotels = get_unique_hotels()
    print(f"Found {len(hotels)} unique hotels.")
    
    # Load existing metadata to resume if interrupted
    metadata = {}
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            
    print(f"Already fetched {len(metadata)} hotels. Resuming...")
    
    count = 0
    # For demonstration, limit to first 10 if not running full batch, or remove limit to process all.
    # We will process all, but you can interrupt with Ctrl+C and it will save progress safely.
    try:
        for hotel in hotels:
            key = f"{hotel['name']}::{hotel['city']}"
            if key in metadata:
                continue
                
            print(f"[{count+1}/{len(hotels)}] Fetching: {hotel['name']} in {hotel['city']}...")
            data = fetch_metadata_nominatim(hotel["name"], hotel["city"])
            
            metadata[key] = {
                "name": hotel["name"],
                "city": hotel["city"],
                "osm_data": data
            }
            
            # Save every 50 records to prevent data loss
            if count % 50 == 0:
                with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                    
            count += 1
            time.sleep(1.2) # Strictly respect Nominatim's 1 req/sec limit
            
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving progress...")
        
    # Final save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        
    print(f"Done. Saved {len(metadata)} hotel metadata records to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
