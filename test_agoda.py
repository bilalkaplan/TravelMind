from curl_cffi import requests
from bs4 import BeautifulSoup
import sys
sys.stdout.reconfigure(encoding='utf-8')

url = "https://www.agoda.com/search?text=Park+Hyatt+Chicago"
response = requests.get(url, impersonate="chrome110")
print(f"Agoda Status: {response.status_code}")
soup = BeautifulSoup(response.text, 'html.parser')
title = soup.title.string if soup.title else 'No Title'
print(f"Title: {title}")
