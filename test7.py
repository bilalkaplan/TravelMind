import json
import sys
import os

# Add src to path
sys.path.append(os.path.abspath('src'))

from cmu_retrieve import search
results = search('Hotel Pennsylvania', location_filter='New York City, NY', top_k_hotels=1)
if results:
    print(f"Buldum: {results[0]['metadata'].get('hotel_name')}")
    print("Olanaklar:", results[0]['metadata'].get('amenities', []))
    print("Oda tipleri:", results[0]['metadata'].get('booking_room_types', []))
    print("Telefon:", results[0]['metadata'].get('phone', ''))
else:
    print('Bulunamadi.')
