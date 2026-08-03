"""Fail-fast verification for TravelMind's local runtime artifacts.

The command is read-only: it does not download or load a model, start a
service, or modify project data. Run it before Streamlit after moving or
rebuilding the project.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


EMBEDDING_REPO_ID = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def fail(message: str) -> None:
    raise SystemExit(f"[PREFLIGHT FAILED] {message}")


def load_nonempty_mapping(path: Path, label: str) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        fail(f"Could not read {label}: {exc}")
    if not isinstance(value, dict) or not value:
        fail(f"{label.capitalize()} is empty or malformed.")
    return value


def verify_embedding_model_cache() -> str:
    """Verify the Hugging Face embedding snapshot without loading it."""
    try:
        from huggingface_hub import scan_cache_dir

        cache = scan_cache_dir()
    except Exception as exc:
        fail(f"Could not inspect the Hugging Face cache: {exc}")

    cached_ids = {
        repo.repo_id
        for repo in cache.repos
        if getattr(repo, "repo_type", None) == "model"
    }
    if EMBEDDING_REPO_ID not in cached_ids:
        fail(
            "The local embedding model is not cached: "
            f"{EMBEDDING_REPO_ID}. Preflight will not download it."
        )
    return EMBEDDING_REPO_ID


def verify_foundry_model_cache() -> tuple[str, str]:
    """Verify the configured Qwen artifact through Foundry's cache command."""
    import config
    from foundry_runtime import find_legacy_foundry_cli, legacy_model_is_cached

    model_id = config.LEGACY_MODEL_ID
    cli = find_legacy_foundry_cli()
    if cli is not None:
        if legacy_model_is_cached(cli, model_id):
            return model_id, "legacy-cache"

        fail(
            f"Configured Foundry model is not cached: {model_id}. "
            "Preflight will not download it."
        )

    # SDK-only installations use the same configured cache root. Checking the
    # files directly avoids SDK initialization and remains fully read-only.
    cache_root = Path(config.FOUNDRY_MODEL_CACHE_DIR)
    slug = model_id.replace(":", "-").casefold()
    candidates = [
        path
        for path in cache_root.glob("*/*")
        if path.is_dir() and path.name.casefold() == slug
    ]
    complete = any(
        any(path.rglob("model.onnx")) and any(path.rglob("model.onnx.data"))
        for path in candidates
    )
    if not complete:
        fail(
            f"Foundry Local is unavailable or its model is not cached: {model_id}. "
            "Preflight will not download it."
        )
    return model_id, "sdk-disk-cache"


def main() -> None:
    data_dir = PROJECT_ROOT / "data"
    required = {
        "database": data_dir / "cmu_travelmind.db",
        "embeddings": data_dir / "embeddings.npy",
        "embedding_ids": data_dir / "embedding_ids.npy",
        "retrieval_row_index": data_dir / "retrieval_row_index.npz",
        "review_stats": data_dir / "hotel_review_stats.json",
        "hotel_metadata": data_dir / "cmu_hotel_metadata.json",
        "rich_hotel_metadata": data_dir / "raw" / "hotel_enriched_raw.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        fail(
            "Missing required local artifacts: " + ", ".join(sorted(missing))
        )

    with sqlite3.connect(required["database"]) as connection:
        try:
            chunk_count, review_chunk_count = connection.execute(
                """SELECT COUNT(*),
                          SUM(CASE WHEN LOWER(chunk_type) LIKE '%review%'
                                   THEN 1 ELSE 0 END)
                     FROM chunks"""
            ).fetchone()
            database_ids = np.fromiter(
                (
                    row[0]
                    for row in connection.execute(
                        "SELECT id FROM chunks ORDER BY id"
                    )
                ),
                dtype=np.int64,
                count=int(chunk_count),
            )
        except sqlite3.Error as exc:
            fail(f"Database schema/read error: {exc}")

    chunk_count = int(chunk_count)
    review_chunk_count = int(review_chunk_count or 0)
    if chunk_count <= 0 or review_chunk_count <= 0:
        fail(
            f"Database has no usable hotel/review chunks: "
            f"chunks={chunk_count}, reviews={review_chunk_count}"
        )

    try:
        matrix = np.load(required["embeddings"], mmap_mode="r")
        embedding_ids = np.load(required["embedding_ids"], mmap_mode="r")
    except (OSError, ValueError) as exc:
        fail(f"Could not read embedding artifacts: {exc}")

    if matrix.ndim != 2 or matrix.shape[0] != chunk_count:
        fail(
            f"Embedding/DB mismatch: matrix={matrix.shape}, chunks={chunk_count}"
        )
    if embedding_ids.ndim != 1 or len(embedding_ids) != chunk_count:
        fail(
            f"Embedding ID/DB mismatch: ids={len(embedding_ids)}, chunks={chunk_count}"
        )
    if not np.array_equal(np.asarray(embedding_ids), database_ids):
        fail("embedding_ids.npy is not aligned with the chunks table.")

    try:
        with np.load(required["retrieval_row_index"], allow_pickle=False) as cache:
            row_index_ids = cache["embedding_ids"]
            row_index_columns = {
                name: cache[name]
                for name in (
                    "chunk_type",
                    "location_norm",
                    "hotel_name_norm",
                    "hotel_id",
                )
            }
    except (OSError, ValueError, KeyError) as exc:
        fail(f"Could not read retrieval_row_index.npz: {exc}")
    if not np.array_equal(row_index_ids, np.asarray(embedding_ids)):
        fail("retrieval_row_index.npz is not aligned with embedding_ids.npy.")
    if any(len(values) != chunk_count for values in row_index_columns.values()):
        fail("retrieval_row_index.npz column length does not match the database.")

    sample_rows = np.linspace(
        0, chunk_count - 1, num=min(chunk_count, 128), dtype=np.int64
    )
    if not np.isfinite(matrix[sample_rows]).all():
        fail("The embedding matrix contains non-finite values in sampled rows.")

    review_stats = load_nonempty_mapping(
        required["review_stats"], "review statistics"
    )
    hotel_metadata = load_nonempty_mapping(
        required["hotel_metadata"], "hotel metadata"
    )
    rich_hotel_metadata = load_nonempty_mapping(
        required["rich_hotel_metadata"], "rich hotel metadata"
    )

    embedding_model = verify_embedding_model_cache()
    model_id, model_cache_source = verify_foundry_model_cache()

    print(
        json.dumps(
            {
                "ok": True,
                "chunks": chunk_count,
                "review_chunks": review_chunk_count,
                "embedding_shape": list(matrix.shape),
                "embedding_ids_aligned": True,
                "retrieval_row_index_aligned": True,
                "review_stat_hotels": len(review_stats),
                "hotel_metadata_records": len(hotel_metadata),
                "rich_hotel_metadata_records": len(rich_hotel_metadata),
                "embedding_model": embedding_model,
                "embedding_model_cached": True,
                "model_id": model_id,
                "model_cached": True,
                "model_cache_source": model_cache_source,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
