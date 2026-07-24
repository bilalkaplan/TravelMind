# -*- coding: utf-8 -*-
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from cmu_rag_answer import generate_conversational_answer

def run_test():
    print("="*60)
    print("TEST 1: ALMANCA (German)")
    print("="*60)
    ans1 = generate_conversational_answer("Guten Tag! Wie geht es dir?", "de", [])
    print(ans1)
    
    print("\n" + "="*60)
    print("TEST 2: ISPANYOLCA (Spanish)")
    print("="*60)
    ans2 = generate_conversational_answer("Hola! Soy un viajero.", "es", [])
    print(ans2)
    
    print("\n" + "="*60)
    print("TEST 3: INGILIZCE (English)")
    print("="*60)
    ans3 = generate_conversational_answer("Who are you and what can you do?", "en", [])
    print(ans3)

if __name__ == '__main__':
    run_test()