import json
import os
import time
import re
from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth

METADATA_FILE = 'data/cmu_hotel_metadata.json'

def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_metadata(metadata):
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def extract_rooms_from_page(page):
    room_types = []
    
    # 1. Try to find application/ld+json schema
    try:
        scripts = page.query_selector_all('script[type="application/ld+json"]')
        for s in scripts:
            try:
                data = json.loads(s.inner_text())
                if isinstance(data, dict) and data.get('@type') in ['Hotel', 'LodgingBusiness']:
                    # sometimes rooms are in "makesOffer" or we just found the hotel
                    pass
            except:
                pass
    except Exception as e:
        print("Schema parsing error:", e)

    # 2. Scrape room names directly from HTML elements
    try:
        # Common classes for room types on booking.com
        selectors = [
            'a.room_link span', 
            'span.room-name', 
            'a[data-room-name="true"]',
            '.room__title',
            '.hprt-roomtype-link'
        ]
        
        for sel in selectors:
            elements = page.query_selector_all(sel)
            for el in elements:
                text = el.inner_text().strip()
                if text and text not in room_types and len(text) > 3:
                    # Clean up things like "Read more"
                    if "read" not in text.lower():
                        room_types.append(text)
                        
        if room_types:
            return room_types
    except Exception as e:
        print("HTML parsing error:", e)
        
    return room_types

def main():
    metadata = load_metadata()
    
    MAX_PROCESSED = 2000 # Process all remaining hotels
    processed = 0
    success = 0
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080}
        )
        
        for hotel_id, hotel_data in metadata.items():
            if processed >= MAX_PROCESSED:
                break
                
            if "booking_room_types" in hotel_data:
                continue
                
            city = hotel_data.get("city", "")
            name = hotel_data.get("name", "")
            
            print(f"Scraping: {name} in {city}")
            
            page = context.new_page()
            Stealth().apply_stealth_sync(page)
            
            try:
                # Search directly via URL
                query = f"{name} {city}"
                search_url = f"https://www.booking.com/searchresults.en-gb.html?ss={query.replace(' ', '+')}"
                
                page.goto(search_url, timeout=30000)
                page.wait_for_load_state("domcontentloaded")
                time.sleep(3) # Wait for anti-bot checks if any
                
                # Check if we got redirected to the hotel page, or if we are still on search results
                if "searchresults" in page.url:
                    first_hotel = page.query_selector('a[data-testid="title-link"]')
                    if first_hotel:
                        href = first_hotel.get_attribute("href")
                        if href:
                            page.goto(href, timeout=30000)
                            page.wait_for_load_state("domcontentloaded")
                            time.sleep(3)
                
                room_types = extract_rooms_from_page(page)
                
                if room_types:
                    hotel_data["booking_room_types"] = room_types
                    success += 1
                    print(f"  -> Found {len(room_types)} room types: {room_types[:2]}...")
                else:
                    # Do not set empty array on failure
                    print("  -> Could not extract room types from page.")
                    
            except Exception as e:
                print(f"  -> Error scraping {name}: {str(e)}")
                # Do not mark as empty array so we can retry later
            finally:
                page.close()
                
            processed += 1
            save_metadata(metadata)
            
            # Sleep to avoid getting IP banned
            import random
            sleep_time = random.uniform(8, 16)
            print(f"  -> Sleeping for {sleep_time:.1f} seconds to avoid IP ban...")
            time.sleep(sleep_time)
            
        browser.close()
        
    print(f"\nFinished scraping. Processed: {processed}, Success: {success}")

if __name__ == "__main__":
    main()
