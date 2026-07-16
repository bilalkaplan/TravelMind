import json
import sqlite3
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"

CHUNKS_PATH = PROCESSED_DIR / "cmu_chunks.jsonl"
DB_PATH = DATA_DIR / "cmu_travelmind.db"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def get_device():
    if torch.cuda.is_available():
        print("Embedding device: CUDA / GPU")
        print("GPU:", torch.cuda.get_device_name(0))
        return "cuda"

    print("Embedding device: CPU")
    return "cpu"


def get_batch_size(device):
    if device == "cuda":
        return 64

    return 32


def load_chunks():
    chunks = []

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Chunk dosyası bulunamadı: {CHUNKS_PATH}")

    with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)

            chunks.append(
                {
                    "chunk_id": item["chunk_id"],
                    "chunk_type": item["chunk_type"],
                    "text": item["text"],
                    "metadata": item.get("metadata", {}),
                }
            )

    return chunks


def create_database():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DB_PATH.exists():
        DB_PATH.unlink()
        print("Eski CMU veritabanı silindi:", DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id TEXT NOT NULL,
            chunk_type TEXT NOT NULL,
            text TEXT NOT NULL,
            metadata_json TEXT,
            embedding_json TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE INDEX idx_chunk_type
        ON chunks(chunk_type)
    """)

    cur.execute("""
        CREATE INDEX idx_chunk_id
        ON chunks(chunk_id)
    """)

    conn.commit()
    return conn


def main():
    print("TravelMind RAG - CMU Vector DB Builder")
    print("-" * 55)

    device = get_device()
    batch_size = get_batch_size(device)

    print("CMU chunk dosyası yükleniyor...")
    chunks = load_chunks()

    print(f"Toplam chunk sayısı: {len(chunks)}")
    print(f"Batch size: {batch_size}")

    print("Embedding modeli yükleniyor...")
    print("Model:", MODEL_NAME)

    model = SentenceTransformer(MODEL_NAME, device=device)

    conn = create_database()
    cur = conn.cursor()

    print("Embedding üretimi başlıyor...")
    print("Bu işlem biraz sürebilir.")

    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))

        batch_chunks = chunks[start:end]
        batch_texts = [chunk["text"] for chunk in batch_chunks]

        embeddings = model.encode(
            batch_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        rows = []

        for chunk, embedding in zip(batch_chunks, embeddings):
            rows.append(
                (
                    chunk["chunk_id"],
                    chunk["chunk_type"],
                    chunk["text"],
                    json.dumps(chunk["metadata"], ensure_ascii=False),
                    json.dumps(embedding.tolist()),
                )
            )

        cur.executemany(
            """
            INSERT INTO chunks (
                chunk_id,
                chunk_type,
                text,
                metadata_json,
                embedding_json
            )
            VALUES (?, ?, ?, ?, ?)
        """,
            rows,
        )

        conn.commit()

        print(f"Kaydedildi: {end} / {len(chunks)}")

    conn.close()

    print("\nCMU vector database oluşturuldu:")
    print(DB_PATH)

    print("\nTamamlandı.")
    print("Embedding device:", device)


if __name__ == "__main__":
    main()
