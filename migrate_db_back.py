import sqlite3
import json
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "cmu_travelmind.db"
EMBEDDINGS_PATH = DATA_DIR / "cmu_travelmind_embeddings.npy"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

try:
    cur.execute("SELECT id, chunk_id, chunk_type, text, metadata_json FROM chunks ORDER BY id")
    rows = cur.fetchall()
except Exception as e:
    print(f"Error: {e}")
    rows = []

if rows:
    print(f"Loaded {len(rows)} rows. Loading numpy array...")
    embeddings_np = np.load(EMBEDDINGS_PATH)
    
    if len(rows) != len(embeddings_np):
        print(f"Mismatch: {len(rows)} rows vs {len(embeddings_np)} embeddings")
    else:
        cur.execute("DROP TABLE IF EXISTS chunks_new")
        cur.execute("""
            CREATE TABLE chunks_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT NOT NULL,
                chunk_type TEXT NOT NULL,
                text TEXT NOT NULL,
                metadata_json TEXT,
                embedding_json TEXT NOT NULL
            )
        """)
        
        insert_rows = []
        for i, r in enumerate(rows):
            emb_list = embeddings_np[i].tolist()
            insert_rows.append((r[0], r[1], r[2], r[3], r[4], json.dumps(emb_list)))
            
        print("Writing restored SQLite table...")
        cur.executemany("INSERT INTO chunks_new (id, chunk_id, chunk_type, text, metadata_json, embedding_json) VALUES (?, ?, ?, ?, ?, ?)", insert_rows)
        
        cur.execute("DROP TABLE chunks")
        cur.execute("ALTER TABLE chunks_new RENAME TO chunks")
        
        cur.execute("CREATE INDEX idx_chunk_type ON chunks(chunk_type)")
        cur.execute("CREATE INDEX idx_chunk_id ON chunks(chunk_id)")
        
        conn.commit()
        conn.execute("VACUUM")
        print("Restore complete!")
conn.close()
