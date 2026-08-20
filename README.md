# TravelMind RAG

TravelMind is a local hotel recommendation and review-question assistant. It
retrieves hotels from the local CMU/TripAdvisor-derived dataset, ranks them,
shows the best matches as Streamlit cards, and keeps the displayed hotels as
grounding for later questions.

The application uses the Qwen model already installed on this computer:
`qwen3-4b-cuda-gpu:2`. Normal startup and the preflight check do **not**
download models.

## What it does

- Ranks hotels for a requested U.S. city and required amenities/room types.
- Shows up to three cards with only recorded facts, including verified
  amenities, room types, hotel class, score, phone, and map link when present.
- Gives Qwen a fact-complete draft of the recommendation so it can express the
  hotel features naturally. A local fact gate rejects invented amenities,
  rooms, prices, links, or numbers and falls back to the grounded draft.
- Tracks the currently selected displayed hotel. A hotel name selects that
  card, “next hotel” advances the selection, and an omitted name initially
  refers to the first card.
- Answers property follow-ups from card metadata and review questions from
  that hotel's retrieved review chunks. For example:
  - `Does Arena Hotel have Wi-Fi and what room types are recorded?`
  - `What do guests say about the rooms there?`
  - `Any complaints about noise?`
  - `How is the service at the second hotel?`
- Handles short conversational turns while keeping unsupported booking,
  price, and non-hotel claims outside the answer.

## Advanced RAG Features (XAI & Hybrid Retrieval)

TravelMind incorporates academic-level enhancements for a mature RAG pipeline:
- **Hybrid Retrieval**: Combines semantic cosine similarity with a fast lexical (BM25-style) keyword scorer to prevent missing exact-match terms.
- **Confidence Transparency**: Missing data signals (e.g., missing hotel class) are explicitly displayed in the UI, fulfilling Explainable AI (XAI) standards.
- **Preference Accumulation**: The Streamlit session state remembers user requirements across multiple conversational turns (e.g., remembering "with pool" when changing cities).
- **Contrastive Explanation**: The system dynamically inserts determinist reasoning into the LLM context, explaining *why* a hotel is ranked lower than another (e.g., due to missing parking data).
- **Ablation Testing**: Includes quantitative test scripts (`scripts/evaluate_ablation.py`) to measure the hallucination-reduction impact of the fact-gate validator.

## Local architecture

- **Retrieval:**
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` runs on CPU
  and searches the normalized local embedding matrix with cosine similarity.
- **Ranking:** the single `WEIGHTS` mapping in
  `src/travelmind_scoring.py` defines the 100-point model. Numeric review
  averages and logarithmically normalized review volume come from
  `data/hotel_review_stats.json`; missing ratings are skipped and the remaining
  weights are renormalized.
- **Generation:** the existing Foundry Local `qwen3-4b-cuda-gpu:2` model uses
  the RTX 3050. Thinking is disabled for user-facing answers, output is bounded
  by `<answer>` tags, and extraction occurs before validation.
- **Review RAG:** review questions search only review chunks belonging to the
  selected hotel. Answers may use the supplied evidence but cannot invent a
  price, reservation state, or link.

MiniLM is deliberately kept on CPU so the laptop's 4 GB VRAM remains available
to Qwen.

## Required local artifacts

The large runtime artifacts are ignored by Git. A clean clone is therefore not
runnable until these existing files/caches have been copied into place:

| Artifact | Purpose |
| --- | --- |
| `data/cmu_travelmind.db` | Hotel and review chunks |
| `data/embeddings.npy` | Normalized chunk embeddings |
| `data/embedding_ids.npy` | Exact database-row alignment for the matrix |
| `data/retrieval_row_index.npz` | Fast startup/filter and per-hotel review memberships |
| `data/hotel_review_stats.json` | Per-hotel numeric review aggregates |
| `data/cmu_hotel_metadata.json` | Base hotel metadata |
| `data/raw/hotel_enriched_raw.json` | Rich amenities and recorded room types |
| Hugging Face cache for MiniLM | Offline query embeddings |
| Foundry Local cache for `qwen3-4b-cuda-gpu:2` | Offline answer generation |

`data/raw/cmu_tripadvisor/review.txt` is required only when rebuilding review
statistics, not for normal chat.

## Install and run on Windows

Python 3.12 is the tested version. From the repository root:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

First run the read-only preflight. It verifies every required data file, exact
database/embedding ID alignment, both local model caches, and never starts or
downloads a model:

```powershell
.venv\Scripts\python.exe scripts\verify_runtime_artifacts.py
```

Then launch Streamlit:

```powershell
.venv\Scripts\python.exe -m streamlit run ui\app.py
```

TravelMind starts the installed Foundry Local service when needed and loads the
already-cached Qwen model into VRAM. Do not run
`scripts\setup_foundry_runtime.py` merely to start the app; that script is only
for explicitly preparing a machine. Its default mode reuses cached artifacts
only; downloads require the separate, explicit `--allow-download` flag.

## Rebuild derived data

After changing the raw reviews:

```powershell
.venv\Scripts\python.exe scripts\build_review_stats.py
```

After changing the database `chunks` table:

```powershell
.venv\Scripts\python.exe scripts\build_embedding_matrix.py
.venv\Scripts\python.exe scripts\build_retrieval_row_index.py
.venv\Scripts\python.exe scripts\verify_runtime_artifacts.py
```

## Regression checks

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts\test_review_questions.py
.venv\Scripts\python.exe scripts\benchmark_detroit.py
```

The Detroit benchmark performs five foreground runs and reports first-token
latency, total time, answer length, preamble extraction, and validator fallback.
The review script exercises the required rooms/noise/service questions and
prints the selected hotel, evidence count, validator messages, and fallback
state.
