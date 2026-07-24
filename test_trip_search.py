from curl_cffi import requests
from bs4 import BeautifulSoup
import sys
sys.stdout.reconfigure(encoding='utf-8')

url = "https://www.trip.com/hotels/list?cityName=Chicago&keyword=Park+Hyatt+Chicago"
response = requests.get(url, impersonate="chrome110")
print(f"Trip.com Status: {response.status_code}")
soup = BeautifulSoup(response.text, 'html.parser')

print(f"Title: {soup.title.string if soup.title else 'No Title'}")
for a in soup.find_all('a', href=True):
    if '/hotels/chicago-hotel' in a['href'] or 'hotel-detail' in a['href']:
        print("Found Hotel Link:", a['href'])
        break
