import json
import os
import time
import requests
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# Set RAPIDAPI_KEY in your environment before running this script; it is
# never read from source so the repo stays safe to publish.
API_KEY = os.environ.get("RAPIDAPI_KEY", "")
HOST = "booking-com15.p.rapidapi.com"
HEADERS = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": HOST
}

METADATA_FILE = 'data/cmu_hotel_metadata.json'

def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_metadata(metadata):
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def search_hotel(hotel_name, city):
    query = f"{hotel_name} {city}"
    url = "https://booking-com15.p.rapidapi.com/api/v1/hotels/searchDestination"
    querystring = {"query": query}
    
    try:
        response = requests.get(url, headers=HEADERS, params=querystring, timeout=10)
        if response.status_code == 429:
            print("Rate limit reached (searchDestination).")
            return None, True
        
        if response.status_code != 200:
            return None, False
            
        data = response.json()
        if "data" in data and isinstance(data["data"], list):
            for item in data["data"]:
                if item.get("search_type") == "hotel":
                    # Check fuzzy match
                    name = item.get("name", "")
                    if similar(name.lower(), hotel_name.lower()) > 0.6:
                        return str(item.get("dest_id")), False
                        
        return None, False
    except Exception as e:
        print(f"Error searching {query}: {e}")
        return None, False

def get_hotel_room_types(hotel_id):
    from datetime import datetime, timedelta
    url = "https://booking-com15.p.rapidapi.com/api/v1/hotels/getHotelDetails"
    checkin = (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d')
    checkout = (datetime.now() + timedelta(days=15)).strftime('%Y-%m-%d')
    
    querystring = {
        "hotel_id": hotel_id,
        "arrival_date": checkin,
        "departure_date": checkout,
        "adults": "1",
        "room_qty": "1"
    }
    
    try:
        response = requests.get(url, headers=HEADERS, params=querystring, timeout=15)
        if response.status_code == 429:
            print("Rate limit reached (getHotelDetails).")
            return None, True
            
        if response.status_code != 200:
            return None, False
            
        data = response.json()
        rooms = data.get("data", {}).get("rooms", {})
        
        room_types = []
        for r_id, room_info in rooms.items():
            r_name = room_info.get("description", "")
            if r_name and r_name not in room_types:
                room_types.append(r_name)
                
        return room_types, False
    except Exception as e:
        print(f"Error getting details for {hotel_id}: {e}")
        return None, False

def main():
    metadata = load_metadata()
    
    # To protect the quota, we set a maximum number of hotels to process at once.
    # RapidAPI usually gives 500 requests/month, but some packages are 50. 2 requests per hotel.
    # For safety, we process at most 20 hotels (40 requests) per run.
    MAX_PROCESSED = 20
    
    processed = 0
    success = 0
    rate_limited = False
    
    for hotel_id, hotel_data in metadata.items():
        if rate_limited or processed >= MAX_PROCESSED:
            break
            
        # Skip if booking data is already pulled
        if "booking_room_types" in hotel_data:
            continue
            
        city = hotel_data.get("city", "")
        name = hotel_data.get("name", "")
        
        print(f"Processing: {name} in {city}")
        b_id, is_rl = search_hotel(name, city)
        
        if is_rl:
            rate_limited = True
            break
            
        if b_id:
            room_types, is_rl = get_hotel_room_types(b_id)
            if is_rl:
                rate_limited = True
                break
                
            if room_types:
                hotel_data["booking_room_types"] = room_types
                success += 1
                print(f"  -> Found {len(room_types)} room types.")
            else:
                hotel_data["booking_room_types"] = []
                print("  -> Found hotel but no specific room types available for dates.")
        else:
            hotel_data["booking_room_types"] = []
            print("  -> Could not find hotel on Booking.")
            
        processed += 1
        save_metadata(metadata)
        
        # Sleep to avoid exceeding the 1 request per second limit
        time.sleep(2)

    print(f"\nFinished processing. Processed: {processed}, Success: {success}")
    if rate_limited:
        print("Stopped due to Rate Limit / Quota.")
    elif processed >= MAX_PROCESSED:
        print(f"Stopped safely after processing {MAX_PROCESSED} hotels to protect your quota.")

if __name__ == "__main__":
    main()
