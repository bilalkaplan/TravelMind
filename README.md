# TravelMind RAG

TravelMind is a local, privacy-first hotel recommendation and review-question assistant using the CMU/TripAdvisor dataset. It leverages an advanced Retrieval-Augmented Generation (RAG) architecture running entirely offline.

## Core Features

- **Hybrid Retrieval**: Merges semantic cosine similarity with lexical BM25 keyword scoring for precise exact-match and contextual search.
- **Explainable AI (XAI)**: UI explicitly details missing data signals (e.g., hotel class) to ensure transparent scoring.
- **Contextual Memory**: Streamlit session state accumulates user preferences across conversational turns.
- **Contrastive Ranking**: Injects deterministic explanations into the LLM context, naturally justifying lower-ranked hotels.
- **Fact-Gate Validator**: A rigorous deterministic interceptor that prevents hallucinations (invented prices, amenities, links). Quantitatively validated via ablation testing.
- **Review RAG**: Answers property follow-up questions purely by grounding against specific guest review chunks.

## Architecture

- **Retrieval**: `paraphrase-multilingual-MiniLM-L12-v2` (CPU) on normalized embeddings.
- **Scoring**: 100-point model using numeric review averages and log-normalized volumes (`src/travelmind_scoring.py`).
- **Generation**: `qwen3-4b-cuda-gpu:2` via Foundry Local on RTX 3050. Enforces `<answer>` tag extraction before validation.

## Local Setup (Windows)

Required artifacts (e.g., `.db`, `.npy`, `.json`) must be present in `data/` as defined in `.gitignore`.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts\verify_runtime_artifacts.py
.venv\Scripts\python.exe -m streamlit run ui\app.py
```

## Testing & Rebuilds

Rebuild local datasets:
```powershell
.venv\Scripts\python.exe scripts\build_review_stats.py
.venv\Scripts\python.exe scripts\build_embedding_matrix.py
.venv\Scripts\python.exe scripts\build_retrieval_row_index.py
```

Run test suite and benchmarks:
```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts\test_review_questions.py
.venv\Scripts\python.exe scripts\benchmark_detroit.py
```
