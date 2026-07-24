import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from cmu_retrieve import search
from cmu_rag_answer import build_hotel_context

def test_hard_filters():
    print("Test 1: Normal Search")
    results1 = search("Hotel Pennsylvania", location_filter="New York", top_k_hotels=1)
    if results1:
        print(build_hotel_context(results1[0], 1))
    
    print("Test 2: Search with 'havuz' filter (Hotel Penn has NO pool)")
    results2 = search("Hotel Pennsylvania havuz", location_filter="New York", top_k_hotels=1)
    if results2:
        print("FAIL: Hotel Penn returned despite hard pool filter")
    else:
        print("SUCCESS: Hotel Penn filtered out because it has no pool.")

    print("Test 3: Search with 'wifi' filter (Hotel Penn HAS wifi)")
    results3 = search("Hotel Pennsylvania wifi", location_filter="New York", top_k_hotels=1)
    if results3:
        print("SUCCESS: Hotel Penn returned because it has wifi.")
        print(build_hotel_context(results3[0], 1))

if __name__ == '__main__':
    test_hard_filters()
