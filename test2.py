import json
with open('data/raw/hotel_enriched_raw.json', 'r', encoding='utf-8') as f:
    d = json.load(f)
print(d.get('W Chicago Lakeshore::Chicago, IL', 'Not found'))
