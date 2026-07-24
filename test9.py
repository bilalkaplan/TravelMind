import sys
import os

# Append src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from cmu_retrieve import search
from cmu_rag_answer import build_hotel_context

def test_context():
    print("Searching for Hotel Pennsylvania in New York...")
    results = search('Hotel Pennsylvania', location_filter='New York City, NY', top_k_hotels=1)
    if results:
        context_str = build_hotel_context(results[0], 1)
        print("\nGenerated Context String:\n")
        print(context_str)
    else:
        print("Not found.")

if __name__ == '__main__':
    test_context()
