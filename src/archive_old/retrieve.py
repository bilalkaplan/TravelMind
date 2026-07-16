import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DB_PATH = Path("data/travelmind.db")
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 5

TURKISH_TO_ENGLISH_HINTS = {
    "temiz": "clean cleanliness hygiene",
    "sessiz": "quiet silent calm",
    "merkezi": "central city center good location",
    "konumu iyi": "good location",
    "çift kişilik": "double bed",
    "çift kişilik yatak": "double bed",
    "tek kişilik": "single bed",
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
}

NON_LOCATION_WORDS = {
    "otel",
    "hotel",
    "hoteli",
    "oteli",
    "temiz",
    "clean",
    "sessiz",
    "quiet",
    "konum",
    "location",
    "merkezi",
    "central",
    "iyi",
    "good",
    "kötü",
    "bad",
    "oda",
    "room",
    "yatak",
    "bed",
    "çift",
    "cift",
    "kişilik",
    "kisilik",
    "double",
    "single",
    "queen",
    "king",
    "arıyorum",
    "ariyorum",
    "istiyorum",
    "looking",
    "want",
    "with",
    "and",
    "for",
    "the",
    "bir",
    "ve",
    "ile",
    "için",
    "icin",
    "puan",
    "rating",
    "score",
    "skor",
    "aile",
    "family",
    "kahvaltı",
    "breakfast",
}


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
        print("Embedding device: CUDA / GPU")
        print("GPU:", torch.cuda.get_device_name(0))
        return device

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


def get_possible_location_tokens(query):
    normalized = normalize_text(query)
    tokens = normalized.split()

    possible_tokens = []

    for token in tokens:
        if len(token) < 3:
            continue

        if token in NON_LOCATION_WORDS:
            continue

        possible_tokens.append(token)

    return possible_tokens


def load_chunks_from_db(chunk_type_filter="hotel"):
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Veritabanı bulunamadı: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT chunk_id, chunk_type, text, metadata_json, embedding_json
        FROM chunks
        WHERE chunk_type = ?
    """,
        (chunk_type_filter,),
    )

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


def calculate_location_score(query, record):
    tokens = get_possible_location_tokens(query)

    if not tokens:
        return 0.0

    metadata = record["metadata"]

    location = normalize_text(metadata.get("location", ""))
    hotel_name = normalize_text(metadata.get("hotel_name", ""))

    if not location:
        return 0.0

    score = 0.0

    for token in tokens:
        if token in location:
            score += 1.0
        elif token in hotel_name:
            score += 0.5

    return score


def search(query, top_k=TOP_K):
    device = get_device()

    print("Embedding modeli yükleniyor...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    print("Hotel kayıtları veritabanından yükleniyor...")
    records, embeddings = load_chunks_from_db("hotel")

    print(f"Toplam hotel kaydı: {len(records)}")

    expanded_query = expand_query(query)

    query_embedding = model.encode(
        expanded_query, normalize_embeddings=True, convert_to_numpy=True
    )

    vector_scores = embeddings @ query_embedding

    location_scores = np.array(
        [calculate_location_score(query, record) for record in records],
        dtype=np.float32,
    )

    location_filter_active = location_scores.max() > 0

    if location_filter_active:
        candidate_indices = np.where(location_scores > 0)[0]
        print(f"Konum filtresi aktif. Eşleşen kayıt sayısı: {len(candidate_indices)}")
    else:
        candidate_indices = np.arange(len(records))
        print("Konum filtresi aktif değil. Tüm hotel kayıtlarında arama yapılıyor.")

    final_scores = []

    for index in candidate_indices:
        vector_score = float(vector_scores[index])
        location_boost = min(float(location_scores[index]) * 0.20, 0.60)
        final_score = vector_score + location_boost

        final_scores.append((index, final_score, vector_score, location_boost))

    final_scores = sorted(final_scores, key=lambda item: item[1], reverse=True)
    top_results = final_scores[:top_k]

    results = []

    for index, final_score, vector_score, location_boost in top_results:
        record = records[index]

        results.append(
            {
                "score": final_score,
                "vector_score": vector_score,
                "location_boost": location_boost,
                "chunk_id": record["chunk_id"],
                "chunk_type": record["chunk_type"],
                "text": record["text"],
                "metadata": record["metadata"],
            }
        )

    return results


def main():
    print("TravelMind RAG - Hotel Retrieval Test")
    print("-" * 45)

    query = input("Otel tercihini ülke/şehir/bölge dahil yaz: ").strip()

    if not query:
        print("Soru boş olamaz.")
        return

    results = search(query)

    print("\nEn alakalı hotel sonuçları:")
    print("=" * 80)

    for i, result in enumerate(results, start=1):
        metadata = result["metadata"]

        print(f"\n{i}. Sonuç")
        print(f"Final skor: {result['score']:.4f}")
        print(f"Vektör skoru: {result['vector_score']:.4f}")
        print(f"Konum boost: {result['location_boost']:.4f}")
        print(f"Chunk ID: {result['chunk_id']}")

        print(f"Otel: {metadata.get('hotel_name', '')}")
        print(f"Konum: {metadata.get('location', '')}")

        hotel_rating = metadata.get("hotel_rating", "")
        room_score = metadata.get("room_score", "")

        if hotel_rating:
            print(f"Otel rating: {hotel_rating} / 10")

        if room_score:
            print(f"Room score: {room_score} / 10")

        print(f"Review count: {metadata.get('review_count', '')}")
        print(f"Oda tipi: {metadata.get('room_type', '')}")
        print(f"Yatak tipi: {metadata.get('bed_type', '')}")
        print(f"Kaynak: {metadata.get('source', '')}")

        print("\nMetin:")
        print(result["text"][:700])
        print("-" * 80)


if __name__ == "__main__":
    main()
