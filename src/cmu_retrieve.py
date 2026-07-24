import json
import re
import sqlite3
import os
from pathlib import Path

# Remove huggingface/transformers environment variables as they are no longer needed
import numpy as np
import foundry_local_sdk as foundry
from foundry_local_sdk import Configuration

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "cmu_travelmind.db"

MODEL_ID = "qwen3-embedding-0.6b-generic-cpu:1"

TOP_K_HOTELS = 8
RAW_CANDIDATE_MULTIPLIER = 8
DEBUG = False

_cached_model = None
_cached_records = None
_cached_embeddings = None


TURKISH_TO_ENGLISH_HINTS = {
    "temiz": "clean cleanliness hygiene spotless",
    "kirli": "dirty unclean filthy",
    "sessiz": "quiet silent calm",
    "gürültülü": "noisy loud",
    "merkezi": "central city center good location",
    "konumu iyi": "good location convenient location central",
    "çift kişilik": "double bed queen king",
    "çift kişilik yatak": "double bed queen king",
    "tek kişilik": "single bed twin",
    "aile": "family",
    "kahvaltı": "breakfast",
    "oda": "room",
    "yatak": "bed",
    "puan": "rating score",
    "skor": "score rating",
    "yorum": "review comment",
    "plaj": "beach",
    "havalimanı": "airport",
    "havuz": "pool swimming pool",
    "wifi": "wifi internet free wifi",
    "internet": "wifi internet",
    "personel": "staff service",
    "servis": "service staff",
    "rahat": "comfortable",
    "konforlu": "comfortable",
    "tekerlekli sandalye": "wheelchair accessible handicap",
    "engelli": "wheelchair accessible handicap disabled",
    "spor": "gym fitness fitness center",
    "egzersiz": "gym fitness fitness center",
    "evcil": "pet friendly pets allowed dog cat",
    "hayvan": "pet friendly pets allowed",
    "köpek": "pet friendly dog",
    "kedi": "pet friendly cat",
}

def normalize_text(text):
    text = str(text).lower()
    text = text.replace("'", " ")
    text = text.replace("’", " ")
    text = re.sub(r"[^a-zA-Z0-9ğüşöçıİĞÜŞÖÇ\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def expand_query(query):
    expanded_parts = [query]
    lower_query = query.lower()
    for turkish_phrase, english_hint in TURKISH_TO_ENGLISH_HINTS.items():
        if turkish_phrase in lower_query:
            expanded_parts.append(english_hint)
    return " ".join(expanded_parts)

def load_chunks_from_db():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"CMU veritabani bulunamadi: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        SELECT chunk_id, chunk_type, text, metadata_json, embedding_json
        FROM chunks
    """)
    rows = cur.fetchall()
    conn.close()

    records = []
    embeddings = []

    for chunk_id, chunk_type, text, metadata_json, embedding_json in rows:
        records.append(
            {
                "chunk_id": chunk_id,
                "chunk_type": chunk_type,
                "text": text,
                "metadata": json.loads(metadata_json),
            }
        )
        embeddings.append(json.loads(embedding_json))

    embeddings = np.array(embeddings, dtype=np.float32)
    return records, embeddings

def get_or_load_embedding_model():
    global _cached_model
    if _cached_model is None:
        if DEBUG:
            print("Foundry Local Embedding modeli yukleniyor...")
        
        # Sadece manager calisiyorsa baslat
        if not hasattr(foundry.FoundryLocalManager, 'instance') or foundry.FoundryLocalManager.instance is None:
            config = Configuration(app_name="TravelMindRAG")
            foundry.FoundryLocalManager.initialize(config)
        
        manager = foundry.FoundryLocalManager.instance
        models = manager.catalog.list_models()
        
        model = manager.catalog.get_model(MODEL_ID)
        if not model:
            for m in models:
                if m.id == MODEL_ID or m.alias == MODEL_ID:
                    model = manager.catalog.get_model(m.alias)
                    break
                    
        if not model:
            raise ValueError(f"Model {MODEL_ID} bulunamadi!")
            
        if not model.is_loaded:
            model.load()
            
        _cached_model = model.get_embedding_client()
    return _cached_model

def get_or_load_chunks():
    global _cached_records, _cached_embeddings
    if _cached_records is None or _cached_embeddings is None:
        if DEBUG:
            print("CMU chunk kayitlari veritabanindan yukleniyor...")
        _cached_records, _cached_embeddings = load_chunks_from_db()
        if DEBUG:
            print(f"Toplam CMU chunk sayisi: {len(_cached_records)}")
    return _cached_records, _cached_embeddings

def extract_total_review_count_from_text(text):
    match = re.search(r"Total review count in CMU dataset:\s*([0-9]+)", str(text))
    if match:
        return match.group(1)
    return ""

def search(query, location_filter=None, filters=None, top_k_hotels=TOP_K_HOTELS):
    model = get_or_load_embedding_model()
    records, embeddings = get_or_load_chunks()

    expanded_query = expand_query(query)

    response = model.generate_embedding(expanded_query)
    query_embedding = np.array(response.data[0].embedding, dtype=np.float32)

    vector_scores = embeddings @ query_embedding

    hard_requirements = {
        'pool': False,
        'wifi': False,
        'breakfast': False,
        'pet': False,
        'gym': False,
        'parking': False,
        'restaurant': False,
        'bar': False,
        'spa': False,
        'room_service': False,
        'business_center': False,
        'tv': False,
        'smoke_free': False
    }
    
    if filters:
        for k, v in filters.items():
            if v and k in hard_requirements:
                hard_requirements[k] = True

    scored_records = []

    for index, record in enumerate(records):
        metadata = record.get("metadata", {})
        if location_filter:
            record_location = normalize_text(metadata.get("location", ""))
            filter_location = normalize_text(location_filter)
            city_part = filter_location.split(',')[0].strip()
            if city_part not in record_location and filter_location not in record_location:
                continue

        failed_req = False
        if any(hard_requirements.values()):
            record_amenities = [str(a).lower() for a in metadata.get("amenities", [])]
            am_str = " ".join(record_amenities)
            if hard_requirements['pool'] and not any(w in am_str for w in ['pool', 'havuz', 'yüzme']):
                failed_req = True
            if hard_requirements['wifi'] and not any(w in am_str for w in ['wifi', 'wi-fi', 'internet', 'kablosuz']):
                failed_req = True
            if hard_requirements['breakfast'] and not any(w in am_str for w in ['breakfast', 'kahvaltı']):
                failed_req = True
            if hard_requirements['pet'] and not any(w in am_str for w in ['pet', 'evcil', 'köpek']):
                failed_req = True
            if hard_requirements['gym'] and not any(w in am_str for w in ['fitness', 'gym', 'spor', 'egzersiz']):
                failed_req = True
            if hard_requirements['parking'] and not any(w in am_str for w in ['parking', 'otopark', 'park']):
                failed_req = True
            if hard_requirements['restaurant'] and not any(w in am_str for w in ['restaurant', 'restoran', 'yemek']):
                failed_req = True
            if hard_requirements['bar'] and not any(w in am_str for w in ['bar', 'lounge']):
                failed_req = True
            if hard_requirements['spa'] and not any(w in am_str for w in ['spa', 'masaj']):
                failed_req = True
            if hard_requirements['room_service'] and not any(w in am_str for w in ['room service', 'oda servisi']):
                failed_req = True
            if hard_requirements['business_center'] and not any(w in am_str for w in ['business center', 'iş merkezi']):
                failed_req = True
            if hard_requirements['tv'] and not any(w in am_str for w in ['tv', 'televizyon']):
                failed_req = True
            if hard_requirements['smoke_free'] and not any(w in am_str for w in ['smoke-free', 'non-smoking', 'sigara içilmez']):
                failed_req = True

        if failed_req:
            continue

        vector_score = float(vector_scores[index])
        chunk_type = record["chunk_type"]

        if chunk_type == "cmu_review_group":
            type_boost = 0.10
        elif chunk_type == "cmu_hotel_profile":
            type_boost = 0.05
        else:
            type_boost = 0.0

        final_score = vector_score + type_boost

        scored_records.append(
            {
                "index": index,
                "score": final_score,
                "vector_score": vector_score,
                "type_boost": type_boost,
                "record": record,
            }
        )

    if not scored_records and location_filter:
        return []

    scored_records = sorted(
        scored_records, key=lambda item: item["score"], reverse=True
    )

    raw_limit = top_k_hotels * RAW_CANDIDATE_MULTIPLIER
    raw_candidates = scored_records[:raw_limit]

    best_by_hotel = {}

    for item in raw_candidates:
        record = item["record"]
        metadata = record["metadata"]

        hotel_id = str(metadata.get("hotel_id", "")).strip()
        hotel_name = str(metadata.get("hotel_name", "")).strip()

        if hotel_id:
            hotel_key = hotel_id
        else:
            hotel_key = normalize_text(hotel_name)

        if not hotel_key:
            continue

        if hotel_key not in best_by_hotel:
            best_by_hotel[hotel_key] = item
            continue

        if item["score"] > best_by_hotel[hotel_key]["score"]:
            best_by_hotel[hotel_key] = item

    unique_results = list(best_by_hotel.values())

    unique_results = sorted(
        unique_results, key=lambda item: item["score"], reverse=True
    )

    results = []

    for item in unique_results[:top_k_hotels]:
        record = item["record"]
        results.append(
            {
                "score": item["score"],
                "vector_score": item["vector_score"],
                "location_boost": item.get("location_boost", 0.0),
                "type_boost": item["type_boost"],
                "chunk_id": record["chunk_id"],
                "chunk_type": record["chunk_type"],
                "text": record["text"],
                "metadata": record["metadata"],
            }
        )

    return results

def main():
    print("TravelMind RAG - CMU Hotel-Level Retrieval Test (Foundry Local)")
    print("-" * 55)
    query = input("Otel tercihini ulke/sehir/bolge dahil yaz: ").strip()
    if not query:
        print("Soru bos olamaz.")
        return

    results = search(query)
    if not results:
        print("Uygun sonuc bulunamadi.")
        return

    print("\nEn alakali tekillestirilmis CMU otel sonuclari:")
    print("=" * 95)
    for i, result in enumerate(results, start=1):
        metadata = result["metadata"]
        text = result["text"]
        total_review_count = metadata.get("review_count_total", "")
        if not total_review_count:
            total_review_count = extract_total_review_count_from_text(text)

        print(f"\n{i}. Otel Sonucu")
        print("-" * 95)
        print(f"Final skor: {result['score']:.4f}")
        print(f"Vektor skoru: {result['vector_score']:.4f}")
        print(f"Otel: {metadata.get('hotel_name', '')}")
        print(f"Konum: {metadata.get('location', '')}")
        print("\nKanit metni:")
        print(text[:1300])
        print("-" * 95)

if __name__ == "__main__":
    main()
