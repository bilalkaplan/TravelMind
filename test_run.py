import sys
import os
sys.path.insert(0, os.path.abspath('src'))
import src.cmu_rag_answer as RAG

inputs = ["Amerika, San Diego'da 3 kişilik otel arıyorum. Konumu çok iyi olsun. Öneride bulunur musun", "çık"]
def mock_input(prompt):
    print(prompt, end='')
    if not inputs:
        return ""
    val = inputs.pop(0)
    print(val)
    return val

import builtins
builtins.input = mock_input

if __name__ == "__main__":
    RAG.main()
