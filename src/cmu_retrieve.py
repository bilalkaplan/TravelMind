import json
import re
import sqlite3
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import logging
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

import numpy as np  # type: ignore # noqa: E402
import torch  # type: ignore # noqa: E402
from sentence_transformers import SentenceTransformer  # type: ignore # noqa: E402

DB_PATH = Path("data/cmu_travelmind.db")
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
    "havuz": "pool",
    "wifi": "wifi internet",
    "internet": "wifi internet",
    "personel": "staff service",
    "servis": "service staff",
    "rahat": "comfortable",
    "konforlu": "comfortable",
}





def get_device():
    forced_device = os.getenv("TRAVELMIND_RETRIEVAL_DEVICE", "").lower().strip()

    if forced_device == "cpu":
        if DEBUG:
            print("Embedding device: CPU zorlandı")
        return "cpu"

    if torch.cuda.is_available():
        if DEBUG:
            print("Embedding device: CUDA / GPU")
            print("GPU:", torch.cuda.get_device_name(0))
        return "cuda"

    if DEBUG:
        print("Embedding device: CPU")
    return "cpu"

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
        raise FileNotFoundError(f"CMU veritabanı bulunamadı: {DB_PATH}")

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


def get_or_load_embedding_model(device):
    global _cached_model
    if _cached_model is None:
        if DEBUG:
            print("Embedding modeli yükleniyor...")
        _cached_model = SentenceTransformer(MODEL_NAME, device=device)
    return _cached_model


def get_or_load_chunks():
    global _cached_records, _cached_embeddings
    if _cached_records is None or _cached_embeddings is None:
        if DEBUG:
            print("CMU chunk kayıtları veritabanından yükleniyor...")
        _cached_records, _cached_embeddings = load_chunks_from_db()
        if DEBUG:
            print(f"Toplam CMU chunk sayısı: {len(_cached_records)}")
    return _cached_records, _cached_embeddings

def extract_total_review_count_from_text(text):
    match = re.search(r"Total review count in CMU dataset:\s*([0-9]+)", str(text))

    if match:
        return match.group(1)

    return ""


def search(query, location_filter=None, top_k_hotels=TOP_K_HOTELS):
    device = get_device()

    model = get_or_load_embedding_model(device)
    records, embeddings = get_or_load_chunks()

    expanded_query = expand_query(query)

    query_embedding = model.encode(
        expanded_query, normalize_embeddings=True, convert_to_numpy=True
    )

    vector_scores = embeddings @ query_embedding

    scored_records = []

    for index, record in enumerate(records):
        # Strict location filtering
        if location_filter:
            record_location = normalize_text(record["metadata"].get("location", ""))
            filter_location = normalize_text(location_filter)
            if filter_location not in record_location:
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
        if DEBUG:
            print("Uyarı: Sorgudaki konumla eşleşen kayıt bulunamadı.")
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
    print("TravelMind RAG - CMU Hotel-Level Retrieval Test")
    print("-" * 55)

    query = input("Otel tercihini ülke/şehir/bölge dahil yaz: ").strip()

    if not query:
        print("Soru boş olamaz.")
        return

    results = search(query)

    if not results:
        print("Uygun sonuç bulunamadı.")
        return

    print("\nEn alakalı tekilleştirilmiş CMU otel sonuçları:")
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
        print(f"Vektör skoru: {result['vector_score']:.4f}")
        print(f"Konum boost: {result['location_boost']:.4f}")
        print(f"Tip boost: {result['type_boost']:.4f}")

        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Chunk type: {result['chunk_type']}")

        print(f"Otel: {metadata.get('hotel_name', '')}")
        print(f"Konum: {metadata.get('location', '')}")
        print(f"Hotel class: {metadata.get('hotel_class', '')}")
        print(f"Total review count: {total_review_count}")
        print(f"Review count in chunk: {metadata.get('review_count_in_chunk', '')}")
        print(f"Kaynak: {metadata.get('source', '')}")

        print("\nKanıt metni:")
        print(text[:1300])
        print("-" * 95)


if __name__ == "__main__":
    main()
