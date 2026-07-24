# -*- coding: utf-8 -*-
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from cmu_rag_answer import generate_conversational_answer
from rich.console import Console
console = Console()

def run_test():
    console.print("\n[bold yellow]--- TEST 1: ALMANCA (German) ---[/bold yellow]")
    ans1 = generate_conversational_answer("Guten Tag! Wie geht es dir?", "de", [])
    console.print(ans1)
    
    console.print("\n[bold yellow]--- TEST 2: İSPANYOLCA (Spanish) ---[/bold yellow]")
    ans2 = generate_conversational_answer("Hola! Soy un viajero.", "es", [])
    console.print(ans2)
    
    console.print("\n[bold yellow]--- TEST 3: İNGİLİZCE (Düşünce Sızıntısı Kontrolü) ---[/bold yellow]")
    ans3 = generate_conversational_answer("Who are you and what can you do?", "en", [])
    console.print(ans3)
    
    console.print("\n[bold green]TÜM TESTLER TAMAMLANDI![/bold green]")

if __name__ == '__main__':
    run_test()