import json

def merge_enriched_data():
    enriched_file = 'data/raw/hotel_enriched_raw.json'
    metadata_file = 'data/cmu_hotel_metadata.json'
    
    with open(enriched_file, 'r', encoding='utf-8') as f:
        enriched_data = json.load(f)
        
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
        
    updated = 0
    for key, info in metadata.items():
        if key in enriched_data:
            edata = enriched_data[key]
            # Update fields
            info['phone'] = edata.get('phone')
            info['osm_tags'] = edata.get('osm_tags', {})
            info['booking_room_types'] = edata.get('room_types', [])
            info['amenities'] = edata.get('amenities', [])
            info['enriched_at'] = edata.get('timestamp')
            updated += 1
            
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        
    print(f'Başarıyla güncellendi: {updated} otel veri tabanına (metadata) aktarıldı.')

if __name__ == '__main__':
    merge_enriched_data()
