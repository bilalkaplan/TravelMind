import json
with open('data/processed/cmu_chunks.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if not line.strip():
            print(f"Empty line at {i}")
            continue
        try:
            json.loads(line)
        except Exception as e:
            print(f"Error at line {i}: {e}")
            print("Content:", repr(line[:100]))
            break
