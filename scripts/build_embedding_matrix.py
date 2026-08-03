"""
One-off build step: reads every embedding_json value from the chunks table
(ordered by id), stacks them into a float32 matrix, L2-normalizes it, and
saves it next to a parallel array of row ids for fast in-process retrieval.

Run manually whenever the chunks table changes:
    .venv\\Scripts\\python.exe scripts\\build_embedding_matrix.py
"""
import json
import sqlite3
import time
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "data" / "cmu_travelmind.db"
EMBEDDINGS_OUT_PATH = ROOT_DIR / "data" / "embeddings.npy"
IDS_OUT_PATH = ROOT_DIR / "data" / "embedding_ids.npy"


def build():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"CMU database not found: {DB_PATH}")

    t0 = time.perf_counter()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM chunks")
    total_rows = cur.fetchone()[0]

    cur.execute("SELECT id, embedding_json FROM chunks ORDER BY id")

    ids = np.empty((total_rows,), dtype=np.int64)
    embeddings = None

    for i, (row_id, embedding_json) in enumerate(cur):
        emb = json.loads(embedding_json)
        if embeddings is None:
            embeddings = np.empty((total_rows, len(emb)), dtype=np.float32)
        embeddings[i] = emb
        ids[i] = row_id

    conn.close()

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = (embeddings / norms).astype(np.float32)

    EMBEDDINGS_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_OUT_PATH, embeddings)
    np.save(IDS_OUT_PATH, ids)

    elapsed = time.perf_counter() - t0
    print(f"Read {total_rows} rows in {elapsed:.2f}s")
    print(f"Saved embeddings matrix {embeddings.shape} (L2-normalized) -> {EMBEDDINGS_OUT_PATH}")
    print(f"Saved ids array {ids.shape} -> {IDS_OUT_PATH}")


if __name__ == "__main__":
    build()
