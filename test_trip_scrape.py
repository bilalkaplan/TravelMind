from duckduckgo_search import DDGS
from curl_cffi import requests
from bs4 import BeautifulSoup
import sys
sys.stdout.reconfigure(encoding='utf-8')

ddgs = DDGS()
hotel_name = "Park Hyatt Chicago"
results = list(ddgs.text(f'site:trip.com/hotels/ {hotel_name}', max_results=2))
print("DDG Results:", results)

if results:
    url = results[0]['href']
    print(f"Fetching: {url}")
    response = requests.get(url, impersonate="chrome110")
    print(f"Status: {response.status_code}")
    soup = BeautifulSoup(response.text, 'html.parser')
    
    room_names = soup.find_all(lambda tag: tag.name in ['h3', 'div', 'span'] and tag.get('class') and any('room' in c.lower() and 'name' in c.lower() for c in tag.get('class')))
    if not room_names:
         for h in soup.find_all(['h2', 'h3']):
             print("Header:", h.text.strip())
    else:
        for r in room_names:
            print("Room:", r.text.strip())
