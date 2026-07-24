import json

with open('data/processed/cmu_chunks.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        chunk = json.loads(line)
        if chunk.get('chunk_type') == 'cmu_hotel_profile' and 'Park Hyatt Chicago' in chunk.get('text', ''):
            print(json.dumps(chunk['metadata'], ensure_ascii=False, indent=2))
            break
