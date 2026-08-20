import json
import re
import sqlite3
import os
import sys
import time
from pathlib import Path

# Chat-time retrieval is fully local. Prevent Hugging Face/Transformers from
# issuing network HEAD requests even when every model file is already cached.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "cmu_travelmind.db"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
EMBEDDING_IDS_PATH = DATA_DIR / "embedding_ids.npy"
ROW_INDEX_PATH = DATA_DIR / "retrieval_row_index.npz"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TOP_K_HOTELS = 8
RAW_CANDIDATE_MULTIPLIER = 8
TOP_N_CANDIDATES = 200
DEBUG = False

_cached_model = None
_cached_matrix = None
_cached_ids = None
_cached_row_index = None


def normalize_text(text):
    text = str(text).lower()
    text = text.replace("'", " ")
    text = text.replace("’", " ")
    text = re.sub(r"[^a-zA-Z0-9ğüşöçıİĞÜŞÖÇ\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def get_or_load_embedding_model():
    global _cached_model
    if _cached_model is None:
        if DEBUG:
            print("Loading SentenceTransformers embedding model (pinned to CPU)...")
        # Pinned to CPU on purpose: the GPU only has 4GB VRAM and that budget
        # is reserved entirely for the LLM (see config.MODEL_ID). System RAM
        # is comparatively plentiful and MiniLM is cheap enough on CPU.
        _cached_model = SentenceTransformer(MODEL_NAME, device="cpu")
    return _cached_model

def get_or_load_matrix():
    """Loads the prebuilt, L2-normalized embedding matrix + aligned row ids.
    Build/refresh it with scripts/build_embedding_matrix.py whenever the
    chunks table changes."""
    global _cached_matrix, _cached_ids
    if _cached_matrix is None or _cached_ids is None:
        if not EMBEDDINGS_PATH.exists() or not EMBEDDING_IDS_PATH.exists():
            raise FileNotFoundError(
                f"Embedding matrix not found ({EMBEDDINGS_PATH}). "
                "Run scripts/build_embedding_matrix.py first."
            )
        t0 = time.perf_counter()
        _cached_matrix = np.load(EMBEDDINGS_PATH)
        _cached_ids = np.load(EMBEDDING_IDS_PATH)
        if DEBUG:
            print(f"Loaded embedding matrix {_cached_matrix.shape} in {time.perf_counter() - t0:.3f}s")
    return _cached_matrix, _cached_ids


def _assemble_row_index(
    chunk_types,
    location_norms,
    hotel_name_norms,
    hotel_ids,
):
    """Build the review memberships from aligned compact row columns."""
    review_indices_by_hotel_id = {}
    review_indices_by_hotel_name = {}
    review_hotel_ids_by_name = {}

    for matrix_position, chunk_type in enumerate(chunk_types):
        if not _is_review_chunk(chunk_type):
            continue
        hotel_id = str(hotel_ids[matrix_position] or "").strip()
        hotel_name_norm = str(hotel_name_norms[matrix_position] or "").strip()
        if hotel_id:
            review_indices_by_hotel_id.setdefault(hotel_id, []).append(
                matrix_position
            )
        if hotel_name_norm:
            review_indices_by_hotel_name.setdefault(
                hotel_name_norm, []
            ).append(matrix_position)
            review_hotel_ids_by_name.setdefault(hotel_name_norm, set()).add(
                hotel_id or f"name:{hotel_name_norm}"
            )

    return {
        "chunk_type": list(chunk_types),
        "location_norm": list(location_norms),
        "hotel_name_norm": list(hotel_name_norms),
        "hotel_id": list(hotel_ids),
        "review_indices_by_hotel_id": {
            key: np.asarray(indices, dtype=np.intp)
            for key, indices in review_indices_by_hotel_id.items()
        },
        "review_indices_by_hotel_name": {
            key: np.asarray(indices, dtype=np.intp)
            for key, indices in review_indices_by_hotel_name.items()
        },
        "review_hotel_ids_by_name": {
            key: frozenset(hotel_id_values)
            for key, hotel_id_values in review_hotel_ids_by_name.items()
        },
    }


def _load_row_index_cache(embedding_ids):
    """Load a safe NumPy-only index when it exactly matches embedding IDs."""
    if not ROW_INDEX_PATH.is_file():
        return None
    try:
        with np.load(ROW_INDEX_PATH, allow_pickle=False) as cache:
            cached_ids = cache["embedding_ids"]
            if not np.array_equal(cached_ids, np.asarray(embedding_ids)):
                return None
            columns = (
                cache["chunk_type"].tolist(),
                cache["location_norm"].tolist(),
                cache["hotel_name_norm"].tolist(),
                cache["hotel_id"].tolist(),
            )
    except (OSError, ValueError, KeyError, AttributeError):
        return None

    if any(len(column) != len(embedding_ids) for column in columns):
        return None
    return _assemble_row_index(*columns)


def write_row_index_cache(row_index=None):
    """Write aligned compact columns; used only by the explicit build script."""
    if row_index is None:
        row_index = get_or_load_row_index()
    _, embedding_ids = get_or_load_matrix()
    ROW_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = ROW_INDEX_PATH.with_suffix(".npz.tmp")
    with temporary_path.open("wb") as target:
        np.savez_compressed(
            target,
            embedding_ids=np.asarray(embedding_ids),
            chunk_type=np.asarray(row_index["chunk_type"], dtype=np.str_),
            location_norm=np.asarray(row_index["location_norm"], dtype=np.str_),
            hotel_name_norm=np.asarray(
                row_index["hotel_name_norm"], dtype=np.str_
            ),
            hotel_id=np.asarray(row_index["hotel_id"], dtype=np.str_),
        )
    temporary_path.replace(ROW_INDEX_PATH)
    return ROW_INDEX_PATH


def get_or_load_row_index():
    """Lightweight per-row index (chunk_type/location/hotel identity only,
    NOT the full text/metadata) used to filter and de-dup candidates without
    touching the database on every query. Built once per process and aligned
    to the embedding-id array by database row id.

    The two ``review_indices_by_*`` mappings are also built here, rather than
    per review question. Their values are embedding-matrix positions and only
    contain review chunks, so a hotel-specific review search cannot leak
    chunks from another hotel or from a hotel profile.
    """
    global _cached_row_index
    if _cached_row_index is None:
        if not DB_PATH.exists():
            raise FileNotFoundError(f"CMU database not found: {DB_PATH}")

        _, embedding_ids = get_or_load_matrix()
        row_count = len(embedding_ids)
        cached_index = _load_row_index_cache(embedding_ids)
        if cached_index is not None:
            _cached_row_index = cached_index
            return _cached_row_index

        matrix_position_by_row_id = {
            int(row_id): position for position, row_id in enumerate(embedding_ids)
        }

        t0 = time.perf_counter()
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT id, chunk_type, metadata_json FROM chunks ORDER BY id")

        chunk_types = [""] * row_count
        location_norms = [""] * row_count
        hotel_name_norms = [""] * row_count
        hotel_ids = [""] * row_count

        for row_id, chunk_type, metadata_json in cur:
            matrix_position = matrix_position_by_row_id.get(int(row_id))
            if matrix_position is None:
                # The database may have changed since the matrix was built.
                # Ignore unembedded rows; the build script will include them
                # the next time the matrix is refreshed.
                continue

            try:
                metadata = json.loads(metadata_json or "{}")
            except (TypeError, ValueError):
                metadata = {}

            hotel_id = str(metadata.get("hotel_id", "") or "").strip()
            hotel_name_norm = normalize_text(
                metadata.get("hotel_name_normalized")
                or metadata.get("hotel_name", "")
            )

            chunk_types[matrix_position] = str(chunk_type or "")
            location_norms[matrix_position] = normalize_text(metadata.get("location", ""))
            hotel_name_norms[matrix_position] = hotel_name_norm
            hotel_ids[matrix_position] = hotel_id

        conn.close()
        _cached_row_index = _assemble_row_index(
            chunk_types,
            location_norms,
            hotel_name_norms,
            hotel_ids,
        )
        if DEBUG:
            print(f"Built row index for {len(chunk_types)} rows in {time.perf_counter() - t0:.3f}s")
    return _cached_row_index


def _is_review_chunk(chunk_type, metadata=None):
    """Accept the review labels used by both the DB and chunk metadata."""
    metadata = metadata or {}
    labels = {
        normalize_text(chunk_type),
        normalize_text(metadata.get("chunk_type", "")),
    }
    return bool(labels & {"cmu review group", "review group", "review"})


def _resolve_hotel_review_indices(hotel_key, row_index):
    """Resolve a card/id/name to the cached review-only matrix positions."""
    id_index = row_index.get("review_indices_by_hotel_id", {})
    name_index = row_index.get("review_indices_by_hotel_name", {})
    hotel_ids_by_name = row_index.get("review_hotel_ids_by_name", {})

    hotel_id = ""
    hotel_name = ""
    if isinstance(hotel_key, dict):
        hotel_id = str(hotel_key.get("hotel_id", "") or "").strip()
        hotel_name = str(hotel_key.get("hotel_name", "") or "").strip()
    elif hotel_key is not None:
        raw_key = str(hotel_key).strip()
        hotel_id = raw_key
        hotel_name = raw_key

    if hotel_id and hotel_id.upper() != "UNKNOWN" and hotel_id in id_index:
        return id_index[hotel_id]

    normalized_name = normalize_text(hotel_name)
    if normalized_name and normalized_name != "unknown":
        exact = name_index.get(normalized_name)
        if exact is not None and len(hotel_ids_by_name.get(normalized_name, ())) <= 1:
            return exact

        # Accept a slightly decorated display name only when it identifies a
        # single cached name. Ambiguous partial names deliberately return no
        # rows instead of mixing evidence from different hotels.
        partial_matches = [
            indices
            for indexed_name, indices in name_index.items()
            if normalized_name in indexed_name or indexed_name in normalized_name
            if len(hotel_ids_by_name.get(indexed_name, ())) <= 1
        ]
        if len(partial_matches) == 1:
            return partial_matches[0]

    return np.empty((0,), dtype=np.intp)

def fetch_chunk_rows_by_id(ids):
    """Fetches full text + metadata_json only for the given row ids."""
    if not ids:
        return {}
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    placeholders = ",".join("?" for _ in ids)
    cur.execute(
        f"SELECT id, chunk_id, chunk_type, text, metadata_json FROM chunks WHERE id IN ({placeholders})",
        [int(i) for i in ids],
    )
    rows = {}
    for row_id, chunk_id, chunk_type, text, metadata_json in cur:
        rows[row_id] = {
            "chunk_id": chunk_id,
            "chunk_type": chunk_type,
            "text": text,
            "metadata": json.loads(metadata_json),
        }
    conn.close()
    return rows


def _rank_review_indices_by_cosine(matrix, candidate_indices, query_embedding, k):
    """Return ``(matrix_position, cosine_score)`` pairs in descending order."""
    query_embedding = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    query_norm = float(np.linalg.norm(query_embedding))
    if not np.isfinite(query_norm) or query_norm == 0.0:
        return []

    candidate_matrix = np.asarray(matrix[candidate_indices], dtype=np.float32)
    if candidate_matrix.ndim != 2 or candidate_matrix.shape[1] != query_embedding.shape[0]:
        raise ValueError(
            "Embedding dimension mismatch between the question and review matrix: "
            f"{query_embedding.shape[0]} != "
            f"{candidate_matrix.shape[1] if candidate_matrix.ndim == 2 else 'invalid'}"
        )

    candidate_norms = np.linalg.norm(candidate_matrix, axis=1)
    denominators = candidate_norms * query_norm
    cosine_scores = np.full((len(candidate_indices),), -np.inf, dtype=np.float32)
    valid = denominators > 0
    cosine_scores[valid] = (
        candidate_matrix[valid] @ query_embedding
    ) / denominators[valid]

    result_count = min(int(k), len(candidate_indices))
    if result_count == len(candidate_indices):
        selected_local = np.arange(len(candidate_indices), dtype=np.intp)
    else:
        selected_local = np.argpartition(cosine_scores, -result_count)[-result_count:]
    selected_local = selected_local[
        np.argsort(cosine_scores[selected_local], kind="stable")[::-1]
    ]

    return [
        (int(candidate_indices[int(local_index)]), float(cosine_scores[int(local_index)]))
        for local_index in selected_local
        if np.isfinite(cosine_scores[int(local_index)])
    ]


def search_reviews_for_hotel(hotel_key, question, k=8):
    """Return the cosine top-k review chunks for one hotel.

    ``hotel_key`` may be a hotel id, a hotel name, or a hotel-card-like dict
    containing ``hotel_id``/``hotel_name``. Candidate membership comes from
    the process-wide row-index cache, so each query scores only this hotel's
    review chunks. Unlike :func:`search`, results are intentionally *not*
    deduplicated by hotel because multiple review excerpts are the evidence
    needed to summarize guest opinion.
    """
    try:
        requested_k = int(k)
    except (TypeError, ValueError):
        requested_k = 8
    if requested_k <= 0:
        return []

    matrix, ids = get_or_load_matrix()
    row_index = get_or_load_row_index()
    candidate_indices = _resolve_hotel_review_indices(hotel_key, row_index)
    if len(candidate_indices) == 0:
        return []

    model = get_or_load_embedding_model()
    query_embedding = model.encode(
        str(question or ""), convert_to_numpy=True
    )
    ranked_candidates = _rank_review_indices_by_cosine(
        matrix, candidate_indices, query_embedding, requested_k
    )
    if not ranked_candidates:
        return []

    selected_row_ids = [int(ids[matrix_index]) for matrix_index, _ in ranked_candidates]
    rows_by_id = fetch_chunk_rows_by_id(selected_row_ids)

    results = []
    for matrix_index, score in ranked_candidates:
        row_id = int(ids[matrix_index])
        record = rows_by_id.get(row_id)
        if record is None or not _is_review_chunk(
            record.get("chunk_type"), record.get("metadata")
        ):
            continue

        results.append(
            {
                "score": score,
                "vector_score": score,
                "chunk_id": record["chunk_id"],
                "chunk_type": record["chunk_type"],
                "text": record["text"],
                "metadata": record["metadata"],
            }
        )

    return results

def extract_total_review_count_from_text(text):
    match = re.search(r"Total review count in CMU dataset:\s*([0-9]+)", str(text))
    if match:
        return match.group(1)
    return ""

RAW_HOTEL_DATA_CACHE = None

def get_full_hotel_metadata(hotel_name, location=None, hotel_id=None):
    """Return the enriched profile for one hotel.

    Hotel names are not globally unique, so a supplied hotel id or location
    is used to disambiguate exact-name matches before falling back to the
    legacy fuzzy lookup.  Callers that only know the name remain supported.
    """
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

    target_lower = str(hotel_name).casefold().strip()
    target_hotel_id = str(hotel_id or "").casefold().strip()

    def normalize(value):
        return re.sub(r"[^a-z0-9]", "", str(value or "").casefold())

    target_location = normalize(location)
    target_city = normalize(str(location or "").split(",", 1)[0])

    def location_matches(key, data):
        if not target_location and not target_city:
            return True
        candidate_values = (
            key,
            data.get("location", ""),
            data.get("city", ""),
        )
        candidate_norms = [normalize(value) for value in candidate_values]
        return any(
            (target_location and target_location in candidate)
            or (target_city and target_city in candidate)
            for candidate in candidate_norms
        )

    exact_name_matches = [
        (key, data)
        for key, data in RAW_HOTEL_DATA_CACHE.items()
        if str(data.get("hotel_name", "")).casefold().strip() == target_lower
    ]

    if target_hotel_id:
        for _key, data in exact_name_matches:
            if str(data.get("hotel_id", "")).casefold().strip() == target_hotel_id:
                return data

    for key, data in exact_name_matches:
        if location_matches(key, data):
            return data
    if exact_name_matches:
        return exact_name_matches[0][1]

    # Legacy fuzzy lookup, still location-aware when the caller supplied one.
    key_matches = [
        (key, data)
        for key, data in RAW_HOTEL_DATA_CACHE.items()
        if target_lower and target_lower in key.casefold()
    ]
    for key, data in key_matches:
        if location_matches(key, data):
            return data
    if key_matches:
        return key_matches[0][1]

    target_norm = normalize(target_lower)
    if target_norm:
        normalized_matches = [
            (key, data)
            for key, data in RAW_HOTEL_DATA_CACHE.items()
            if target_norm in normalize(key)
        ]
        for key, data in normalized_matches:
            if location_matches(key, data):
                return data
        if normalized_matches:
            return normalized_matches[0][1]

    return {}

def search(query, location_filter=None, filters=None, top_k_hotels=TOP_K_HOTELS, requested_hotel_name=None):
    model = get_or_load_embedding_model()
    matrix, ids = get_or_load_matrix()
    row_index = get_or_load_row_index()

    query_embedding = model.encode(query, convert_to_numpy=True).astype(np.float32)

    # Single vectorized matrix multiplication over the full corpus.
    vector_scores = matrix @ query_embedding

    n_rows = vector_scores.shape[0]
    mask = np.ones(n_rows, dtype=bool)

    req_hotel_norm = normalize_text(requested_hotel_name) if requested_hotel_name else None

    filter_location = normalize_text(location_filter) if location_filter else None
    city_part = filter_location.split(',')[0].strip() if filter_location else None

    location_norm_list = row_index["location_norm"]
    hotel_name_norm_list = row_index["hotel_name_norm"]
    chunk_type_list = row_index["chunk_type"]

    if location_filter:
        mask &= np.fromiter(
            (city_part in loc or filter_location in loc for loc in location_norm_list),
            dtype=bool,
            count=n_rows,
        )

    if req_hotel_norm:
        mask &= np.fromiter(
            (req_hotel_norm in hn or hn in req_hotel_norm for hn in hotel_name_norm_list),
            dtype=bool,
            count=n_rows,
        )

    if not mask.any():
        return []

    masked_scores = np.where(mask, vector_scores, -np.inf)

    # Top-N via argpartition instead of a full O(n log n) sort.
    n_candidates = min(TOP_N_CANDIDATES, n_rows)
    top_idx = np.argpartition(masked_scores, -n_candidates)[-n_candidates:]
    top_idx = top_idx[np.isfinite(masked_scores[top_idx])]

    # Metadata/text is fetched from the database only for this small candidate set.
    candidate_rows = fetch_chunk_rows_by_id([int(ids[i]) for i in top_idx])

    query_norm = normalize_text(query)
    query_terms = set(query_norm.split())
    stop_words = {"hotel", "hotels", "in", "the", "a", "an", "and", "or", "with", "for", "to", "of", "is", "are"}
    keyword_terms = query_terms - stop_words

    scored_records = []
    for i in top_idx:
        row_id = int(ids[i])
        row = candidate_rows.get(row_id)
        if row is None:
            continue

        vector_score = float(vector_scores[i])
        chunk_type = chunk_type_list[i]

        if chunk_type == "cmu_review_group":
            type_boost = 0.10
        elif chunk_type == "cmu_hotel_profile":
            type_boost = 0.05
        else:
            type_boost = 0.0

        # Hybrid Retrieval (Vector + Keyword)
        keyword_score = 0.0
        if keyword_terms:
            text_norm = normalize_text(row["text"])
            hotel_name_norm = normalize_text(row["metadata"].get("hotel_name", ""))
            
            matches = 0
            for term in keyword_terms:
                if f" {term} " in f" {text_norm} ":
                    matches += 1
                if term in hotel_name_norm:
                    matches += 2
                    
            keyword_score = (matches / len(keyword_terms)) * 0.15

        final_score = vector_score + type_boost + keyword_score

        scored_records.append(
            {
                "score": final_score,
                "vector_score": vector_score,
                "type_boost": type_boost,
                "keyword_score": keyword_score,
                "record": row,
            }
        )

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
                "keyword_score": item.get("keyword_score", 0.0),
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
    query = input("Enter your hotel preference including country/city/region: ").strip()
    if not query:
        print("Query cannot be empty.")
        return

    results = search(query)
    if not results:
        print("No suitable result found.")
        return

    print("\nMost relevant deduplicated CMU hotel results:")
    print("=" * 95)
    for i, result in enumerate(results, start=1):
        metadata = result["metadata"]
        text = result["text"]
        total_review_count = metadata.get("review_count_total", "")
        if not total_review_count:
            total_review_count = extract_total_review_count_from_text(text)

        print(f"\n{i}. Hotel Result")
        print("-" * 95)
        print(f"Final score: {result['score']:.4f}")
        print(f"Vector score: {result['vector_score']:.4f}")
        print(f"Hotel: {metadata.get('hotel_name', '')}")
        print(f"Location: {metadata.get('location', '')}")
        print("\nEvidence text:")
        print(text[:1300])
        print("-" * 95)

if __name__ == "__main__":
    main()
