import json
import sqlite3
from pathlib import Path
import sys

# Replace sentence_transformers with foundry_local_sdk
import foundry_local_sdk as foundry
from foundry_local_sdk import Configuration

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"

CHUNKS_PATH = PROCESSED_DIR / "cmu_chunks.jsonl"
DB_PATH = DATA_DIR / "cmu_travelmind.db"

MODEL_ID = "qwen3-embedding-0.6b-generic-cpu:1"

def init_foundry_model():
    print(f"Foundry SDK baslatiliyor...")
    config = Configuration(app_name="TravelMindRAG")
    foundry.FoundryLocalManager.initialize(config)
    manager = foundry.FoundryLocalManager.instance

    # Models
    models = manager.catalog.list_models()
    model = manager.catalog.get_model(MODEL_ID)
    
    if not model:
        # Try to find by alias if alias matches
        for m in models:
            if m.id == MODEL_ID or m.alias == MODEL_ID:
                model = manager.catalog.get_model(m.alias)
                break

    if not model:
        raise ValueError(f"Model {MODEL_ID} bulunamadi! Lutfen once modeli Foundry Local uzerinden indirin.")

    print(f"Model {model.alias} hazirlaniyor...")
    if not model.is_cached:
        print("Model indiriliyor (Bu islem bir defaya mahsus zaman alabilir)...")
        model.download()
    
    if not model.is_loaded:
        print("Model yukleniyor...")
        model.load()

    client = model.get_embedding_client()
    return client


def load_chunks():
    chunks = []
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Chunk dosyasi bulunamadi: {CHUNKS_PATH}")

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
        print("Eski CMU veritabani silindi:", DB_PATH)

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

    cur.execute("CREATE INDEX idx_chunk_type ON chunks(chunk_type)")
    cur.execute("CREATE INDEX idx_chunk_id ON chunks(chunk_id)")

    conn.commit()
    return conn


def main():
    print("TravelMind RAG - CMU Vector DB Builder (Foundry Local)")
    print("-" * 55)

    print("CMU chunk dosyasi yukleniyor...")
    chunks = load_chunks()
    print(f"Toplam chunk sayisi: {len(chunks)}")

    client = init_foundry_model()

    conn = create_database()
    cur = conn.cursor()

    print("Embedding uretimi basliyor...")
    
    batch_size = 16
    
    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        batch_chunks = chunks[start:end]
        batch_texts = [chunk["text"] for chunk in batch_chunks]

        # Get embeddings from Foundry Local SDK
        response = client.generate_embeddings(batch_texts)
        
        rows = []
        for i, chunk in enumerate(batch_chunks):
            embedding_vector = response.data[i].embedding
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
        print(f"Kaydedildi: {end} / {len(chunks)}", flush=True)

    conn.close()
    print(f"\nCMU vector database olusturuldu:\n{DB_PATH}\nTamamlandi.")

if __name__ == "__main__":
    main()
