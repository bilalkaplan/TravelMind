import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from cmu_retrieve import search
from cmu_rag_answer import build_hotel_context

def test_debug():
    print("Testing hard filter debug...")
    results = search("Hotel Pennsylvania havuz", location_filter="New York", top_k_hotels=1)
    for r in results:
        print("Returned:", r['metadata']['hotel_name'])
        print("Amenities:", r['metadata'].get('amenities', []))
        print("OSM:", r['metadata'].get('osm_tags', {}))

if __name__ == '__main__':
    test_debug()
