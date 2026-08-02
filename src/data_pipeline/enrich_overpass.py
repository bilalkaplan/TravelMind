import json
import time
import requests
import difflib
from pathlib import Path

DATA_DIR = Path("data")
OUTPUT_PATH = DATA_DIR / "cmu_hotel_metadata.json"

def clean_city_name(city):
    # E.g. 'New York City, NY' -> 'New York' (Overpass usually prefers the short name)
    return city.split(",")[0].strip()

def query_overpass(city_name):
    print(f"Querying Overpass API for all hotels in: {city_name}...")
    # Using a bounding area based on name. Sometimes 'name' is ambiguous (e.g. 'Dallas').
    # Adding network timeout and retries.
    
    # Overpass QL to get nodes and ways tagged with tourism=hotel in the city area
    overpass_url = "https://overpass-api.de/api/interpreter"
    # We use search by name in US to constrain area. 
    overpass_query = f"""
    [out:json][timeout:25];
    area["name"="{city_name}"]["admin_level"~"4|8"]->.searchArea;
    (
      node["tourism"="hotel"](area.searchArea);
      way["tourism"="hotel"](area.searchArea);
    );
    out tags;
    """
    
    headers = {'User-Agent': 'TravelMind-RAG-Internship-Project/1.0 (contact: test@example.com)'}
    try:
        response = requests.post(overpass_url, data={'data': overpass_query}, headers=headers, timeout=30)
        if response.status_code == 200:
            return response.json().get('elements', [])
        elif response.status_code == 429:
            print("Rate limited by Overpass. Waiting 30 seconds...")
            time.sleep(30)
            return query_overpass(city_name)
        else:
            print(f"Failed to query {city_name}. Status: {response.status_code}")
    except Exception as e:
        print(f"Error querying {city_name}: {e}")
    return []

def main():
    if not OUTPUT_PATH.exists():
        print("Error: cmu_hotel_metadata.json not found.")
        return

    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Group hotels by city
    hotels_by_city = {}
    for key, hotel in metadata.items():
        city = hotel.get('city')
        if city:
            c = clean_city_name(city)
            if c not in hotels_by_city:
                hotels_by_city[c] = []
            hotels_by_city[c].append(key)

    total_matched = 0
    
    for city, keys in hotels_by_city.items():
        # Check if any hotel in this city already has non-empty osm_tags
        has_osm = any(metadata[k].get("osm_tags") for k in keys)
        if has_osm:
            print(f"Skipping {city} as it already has some osm_tags.")
            continue

        osm_hotels = query_overpass(city)
        if not osm_hotels:
            # Overpass can be tricky with area names, fallback to a simpler or skip
            continue
            
        print(f"Found {len(osm_hotels)} hotels in OSM for {city}. Matching against our {len(keys)} hotels...")
        
        # Build a dict of {osm_name_lower: tags}
        osm_name_dict = {}
        for elem in osm_hotels:
            tags = elem.get('tags', {})
            name = tags.get('name')
            if name:
                osm_name_dict[name.lower()] = tags
        
        osm_names = list(osm_name_dict.keys())
        if not osm_names:
            continue
            
        matched_in_city = 0
        for key in keys:
            hotel = metadata[key]
            our_name = hotel['name'].lower()
            
            # Fuzzy match
            matches = difflib.get_close_matches(our_name, osm_names, n=1, cutoff=0.6)
            if matches:
                best_match = matches[0]
                tags = osm_name_dict[best_match]
                
                # Extract valuable tags
                extracted = {
                    "breakfast": tags.get("breakfast"),
                    "internet_access": tags.get("internet_access", tags.get("wifi")),
                    "wheelchair": tags.get("wheelchair"),
                    "swimming_pool": tags.get("swimming_pool"),
                    "stars": tags.get("stars"),
                    "rooms": tags.get("rooms"),
                    "beds": tags.get("beds")
                }
                # Clean None values
                extracted = {k: v for k, v in extracted.items() if v is not None}
                
                if extracted:
                    metadata[key]["osm_tags"] = extracted
                    matched_in_city += 1
                    total_matched += 1
        
        print(f"Successfully matched {matched_in_city}/{len(keys)} hotels in {city}.")
        
        # Save back incrementally
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            
        time.sleep(2) # be nice to overpass

    print(f"Total hotels enriched with Overpass data: {total_matched}/{len(metadata)}")

if __name__ == "__main__":
    main()
