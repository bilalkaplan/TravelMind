from curl_cffi import requests
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

url = "https://www.booking.com/graphql"
headers = {
    "content-type": "application/json",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
    "x-booking-graphql-client-id": "1", # usually some ID is needed
}
query = """
query {
  searchQueries {
    search(
      request: {
        destinations: [{ text: "Park Hyatt Chicago" }]
      }
    ) {
      properties {
        name
        blocks {
          roomName
        }
      }
    }
  }
}
"""
try:
    response = requests.post(url, headers=headers, json={"query": query}, impersonate="chrome110")
    print(f"Status: {response.status_code}")
    print(response.text[:500])
except Exception as e:
    print(f"Error: {e}")
