import json
import time
import requests
import os
import subprocess
import traceback

DATA_FILE = 'data/cmu_hotel_metadata.json'

def fetch_osm_data(hotel_name, location):
    url = "https://nominatim.openstreetmap.org/search"
    query = f"{hotel_name} {location}"
    params = {
        "q": query,
        "format": "json",
        "extratags": 1,
        "addressdetails": 1,
        "limit": 1
    }
    headers = {
        "User-Agent": "TravelMind-RAG-Update/1.0 (test@example.com)"
    }
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                return data[0].get('extratags', {})
    except Exception as e:
        print(f"Error fetching OSM for {hotel_name}: {e}")
    return {}

def main():
    print("Starting OSM Data Enrichment...")
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        hotels = json.load(f)

    total = len(hotels)
    processed = 0
    updated = 0

    for idx, (hotel_id, hotel_data) in enumerate(hotels.items()):
        # Skip if already has osm_tags
        if hotel_data.get('osm_tags'):
            processed += 1
            continue

        hotel_name = hotel_data.get('name', '')
        location = hotel_data.get('city', '')
        
        if not hotel_name:
            continue

        print(f"[{idx+1}/{total}] Fetching OSM for: {hotel_name}")
        osm_tags = fetch_osm_data(hotel_name, location)
        
        hotel_data['osm_tags'] = osm_tags
        updated += 1
        processed += 1

        # Save progress every 20 hotels
        if updated % 20 == 0:
            with open(DATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(hotels, f, ensure_ascii=False, indent=4)
            print("Progress saved.")

        # Respect Nominatim rate limit: max 1 request per second
        time.sleep(1.1)

    # Final save
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(hotels, f, ensure_ascii=False, indent=4)
    
    print(f"OSM Enrichment Complete! Updated {updated} hotels.")
    
    # Run the embedding script
    print("Starting FAISS Embedding DB recreation...")
    try:
        import sys
        print("Creating chunks...")
        subprocess.run([sys.executable, "src/create_cmu_chunks.py"], check=True)
        print("Building vector database...")
        subprocess.run([sys.executable, "src/build_cmu_vector_db.py"], check=True)
        print("Embedding DB recreated successfully!")
    except subprocess.CalledProcessError as e:
        print("Error recreating embeddings!")
        print(e.stderr)
    except Exception as e:
        import sys
        try:
            result = subprocess.run([sys.executable, "src/create_cmu_chunks.py"], check=True, capture_output=True, text=True)
            print("Embedding DB recreated successfully!")
            print(result.stdout)
        except Exception as e2:
             print("Failed to run embedding script:", e2)

if __name__ == "__main__":
    main()
