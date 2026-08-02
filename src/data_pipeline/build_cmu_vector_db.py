import json
import sqlite3
from pathlib import Path
import sys
import torch
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"

CHUNKS_PATH = PROCESSED_DIR / "cmu_chunks.jsonl"
DB_PATH = DATA_DIR / "cmu_travelmind.db"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def get_device():
    if torch.cuda.is_available():
        device = "cuda"
        print("Embedding device: CUDA / GPU")
        print("GPU:", torch.cuda.get_device_name(0))
        return device
    print("Embedding device: CPU")
    return "cpu"

def split_text_into_chunks(text, max_chars=1500):
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    while len(text) > max_chars:
        split_idx = text.rfind(' ', 0, max_chars)
        if split_idx == -1:
            split_idx = max_chars
        chunks.append(text[:split_idx].strip())
        text = text[split_idx:].strip()
    if text:
        chunks.append(text)
    return chunks

def load_chunks():
    chunks = []
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Chunk dosyasi bulunamadi: {CHUNKS_PATH}")

    with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            text_parts = split_text_into_chunks(item["text"], max_chars=1500)
            
            if len(text_parts) == 1:
                chunks.append(
                    {
                        "chunk_id": item["chunk_id"],
                        "chunk_type": item["chunk_type"],
                        "text": text_parts[0],
                        "metadata": item.get("metadata", {}),
                    }
                )
            else:
                for i, part in enumerate(text_parts):
                    chunks.append(
                        {
                            "chunk_id": f"{item['chunk_id']}_{i+1}",
                            "chunk_type": item["chunk_type"],
                            "text": part,
                            "metadata": item.get("metadata", {}),
                        }
                    )
    return chunks

def create_database():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id TEXT NOT NULL,
            chunk_type TEXT NOT NULL,
            text TEXT NOT NULL,
            metadata_json TEXT,
            embedding_json TEXT NOT NULL
        )
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunk_type ON chunks(chunk_type)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunk_id ON chunks(chunk_id)")
    conn.commit()
    return conn

def main():
    print("TravelMind RAG - CMU Vector DB Builder (SentenceTransformers - MiniLM)")
    print("-" * 55)

    print("CMU chunk dosyasi yukleniyor...")
    chunks = load_chunks()
    print(f"Toplam chunk sayisi: {len(chunks)}")

    device = get_device()
    print(f"Model yukleniyor: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device=device)

    conn = create_database()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM chunks")
    saved_count = cur.fetchone()[0]
    print(f"Veritabaninda hazir bulunan kayit sayisi: {saved_count}")

    print("Embedding uretimi basliyor...")
    
    batch_size = 128 if device == "cuda" else 64
    print(f"Batch size: {batch_size}")
    
    import time
    chunks_to_process = chunks[saved_count:]
    processed_count = saved_count

    for start in range(0, len(chunks_to_process), batch_size):
        end = min(start + batch_size, len(chunks_to_process))
        batch_chunks = chunks_to_process[start:end]
        batch_texts = [chunk["text"] for chunk in batch_chunks]

        # Generate embeddings directly using SentenceTransformers
        embeddings = model.encode(batch_texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
        
        rows = []
        for i, chunk in enumerate(batch_chunks):
            embedding_vector = embeddings[i].tolist()
            rows.append(
                (
                    chunk["chunk_id"],
                    chunk["chunk_type"],
                    chunk["text"],
                    json.dumps(chunk["metadata"], ensure_ascii=False),
                    json.dumps(embedding_vector),
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
        processed_count += len(batch_chunks)
        print(f"Kaydedildi: {processed_count} / {len(chunks)}", flush=True)

    conn.close()
    print(f"\nCMU vector database olusturuldu:\n{DB_PATH}\nTamamlandi.")

if __name__ == "__main__":
    main()
