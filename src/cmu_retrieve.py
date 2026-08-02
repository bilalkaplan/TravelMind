import json
import re
import sqlite3
import os
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "cmu_travelmind.db"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

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

    cur.execute("SELECT COUNT(*) FROM chunks")
    total_rows = cur.fetchone()[0]

    cur.execute("""
        SELECT chunk_id, chunk_type, text, metadata_json, embedding_json
        FROM chunks
    """)

    records = []
    embeddings = None

    for i, (chunk_id, chunk_type, text, metadata_json, embedding_json) in enumerate(cur):
        records.append(
            {
                "chunk_id": chunk_id,
                "chunk_type": chunk_type,
                "text": text,
                "metadata": json.loads(metadata_json),
            }
        )
        emb_list = json.loads(embedding_json)
        if embeddings is None:
            embeddings = np.empty((total_rows, len(emb_list)), dtype=np.float32)
        embeddings[i] = emb_list
        
    conn.close()

    return records, embeddings

def get_or_load_embedding_model():
    global _cached_model
    if _cached_model is None:
        if DEBUG:
            print("SentenceTransformers Embedding modeli yukleniyor...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _cached_model = SentenceTransformer(MODEL_NAME, device=device)
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

RAW_HOTEL_DATA_CACHE = None

def get_full_hotel_metadata(hotel_name):
    global RAW_HOTEL_DATA_CACHE
    if RAW_HOTEL_DATA_CACHE is None or not RAW_HOTEL_DATA_CACHE:
        try:
            import json
            import os
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            json_path = os.path.join(base_dir, 'data', 'raw', 'hotel_enriched_raw.json')
            with open(json_path, 'r', encoding='utf-8') as f:
                RAW_HOTEL_DATA_CACHE = json.load(f)
        except Exception as e:
            print(f"Error loading metadata JSON: {e}")
            RAW_HOTEL_DATA_CACHE = {}

    target_lower = str(hotel_name).lower().strip()
    
    # 1. Exact match on hotel_name field
    for key, data in RAW_HOTEL_DATA_CACHE.items():
        if str(data.get("hotel_name", "")).lower().strip() == target_lower:
            return data
            
    # 2. Substring match on key
    for key, data in RAW_HOTEL_DATA_CACHE.items():
        if target_lower in key.lower():
            return data

    # 3. Robust Alphanumeric fallback match
    import re
    target_norm = re.sub(r'[^a-z0-9]', '', target_lower)
    if target_norm:
        for key, data in RAW_HOTEL_DATA_CACHE.items():
            if target_norm in re.sub(r'[^a-z0-9]', '', key.lower()):
                return data

    return {}

def search(query, location_filter=None, filters=None, top_k_hotels=TOP_K_HOTELS, requested_hotel_name=None):
    model = get_or_load_embedding_model()
    records, embeddings = get_or_load_chunks()

    expanded_query = expand_query(query)

    query_embedding = model.encode(expanded_query, convert_to_numpy=True).astype(np.float32)

    # We no longer hard-filter based on requirements here.
    # Instead, we retrieve the top candidates and apply ranking penalties 
    # for missing or unknown requirements in hotel_card_builder.py.
    vector_scores = embeddings @ query_embedding

    scored_records = []

    req_hotel_norm = normalize_text(requested_hotel_name) if requested_hotel_name else None

    filter_location = normalize_text(location_filter) if location_filter else None
    city_part = filter_location.split(',')[0].strip() if filter_location else None

    for index, record in enumerate(records):
        metadata = record.get("metadata", {})
        if location_filter:
            record_location = normalize_text(metadata.get("location", ""))
            if city_part not in record_location and filter_location not in record_location:
                continue

        if req_hotel_norm:
            rec_hotel_norm = normalize_text(metadata.get("hotel_name", ""))
            # Fuzzy match or subset match for the hotel name
            if req_hotel_norm not in rec_hotel_norm and rec_hotel_norm not in req_hotel_norm:
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
