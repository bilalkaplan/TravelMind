import json
import math

METADATA_FILE = 'data/cmu_hotel_metadata.json'

try:
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    total = len(data)
    done = sum(1 for v in data.values() if 'booking_room_types' in v)
    
    if total == 0:
        print("Veritabanı boş.")
    else:
        percent = (done / total) * 100
        
        # 50 karakterlik progress bar
        bar_len = 50
        filled_len = math.floor(bar_len * done / total)
        bar = '#' * filled_len + '-' * (bar_len - filled_len)
        
        print("\n=== Booking.com Veri Çekme İlerlemesi ===")
        print(f"İlerleme: |{bar}| {percent:.2f}%")
        print(f"Durum:    {done} / {total} Otel Tamamlandı\n")
        
except Exception as e:
    print("Hata:", e)
