"""Build TravelMind's compact, aligned retrieval row index.

The resulting NumPy archive contains no review text or Python pickle objects.
It removes the need to parse 224k metadata JSON rows every time Streamlit
starts. Rebuild it whenever the chunks table or embedding IDs change.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cmu_retrieve import get_or_load_row_index, write_row_index_cache


def main() -> None:
    started = time.perf_counter()
    row_index = get_or_load_row_index()
    output_path = write_row_index_cache(row_index)
    elapsed = time.perf_counter() - started
    print(f"Rows: {len(row_index['chunk_type'])}")
    print(f"Review hotels: {len(row_index['review_indices_by_hotel_id'])}")
    print(f"Output: {output_path}")
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
