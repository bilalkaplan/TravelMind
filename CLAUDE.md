# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

TravelMind is a locally-hosted RAG hotel-recommendation assistant. It only answers hotel/accommodation questions for a fixed set of ~25 US cities, using a CMU/TripAdvisor-derived dataset of hotels and reviews. It intentionally does not do flights, visas, itineraries, or live pricing/booking. The UI and all model output are English-only (`ui/app.py` keeps a fixed English string dictionary, not a language switcher).

Two local models power it, both chosen to avoid network calls and API costs:
- **Embeddings**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (PyTorch, local) for retrieval over precomputed chunk embeddings.
- **Generation**: a local LLM served by **Foundry Local** (`foundry service start`), accessed through the `openai` SDK pointed at the discovered local endpoint — no API key, no external network calls at chat time. The runtime alias is `qwen3-4b` (`config.MODEL_ALIAS`); `config.LEGACY_MODEL_ID` (`qwen3-4b-cuda-gpu:2`) is a retired CLI-catalog ID kept only as a fallback lookup, not the model actually served — don't describe it as "the model" in docs.

## Commands

Python env lives in `.venv/` (Windows), Python 3.12. `requirements.txt` is a curated, UTF-8, runtime-only dependency list (add a package with `pip install` into `.venv`, then reflect it here by hand — don't overwrite this file with a raw `pip freeze`, which would also pull in offline/data-pipeline-only and dev-only packages like `playwright`/`pandas` that are intentionally excluded).

```powershell
# Run the full backend test suite
.venv\Scripts\python.exe -m pytest tests\ -q

# Run a single test file / test
.venv\Scripts\python.exe -m pytest tests\test_scoring.py -q
.venv\Scripts\python.exe -m pytest tests\test_router.py::test_router_dirty_json -q

# Start the local LLM service (separate terminal, required before chatting)
foundry service start

# Verify local data artifacts are present and consistent (no network calls)
.venv\Scripts\python.exe scripts\verify_runtime_artifacts.py

# Launch the Streamlit UI
.venv\Scripts\python.exe -m streamlit run ui\app.py

# Backend-only smoke/regression checks
.venv\Scripts\python.exe scripts\smoke_test_travelmind.py
.venv\Scripts\python.exe scripts\test_review_questions.py

# Quantitative fact-gate ablation (needs Foundry running; see Validation below)
.venv\Scripts\python.exe scripts\evaluate_ablation.py

# Detroit latency/fallback-rate benchmark (needs Foundry running)
.venv\Scripts\python.exe scripts\benchmark_detroit.py
```

`src/` is not a package (no `__init__.py`); every entry point (`ui/app.py`, `tests/conftest.py`, `scripts/*.py`) manually does `sys.path.insert/append` to make `src` modules importable. New modules under `src/` should follow the existing flat-import style (`from cmu_retrieve import search`, not package-relative imports).

## Architecture

The core design is a **layered anti-hallucination pipeline** — the LLM is never trusted to know or invent hotel facts; it only narrates data that was already verified in code:

1. **Retrieval** — `src/cmu_retrieve.py`: loads a precomputed embedding matrix (`data/embeddings.npy` + `data/embedding_ids.npy`, with `data/retrieval_row_index.npz` for fast per-hotel/review lookups) rather than reading `embedding_json` per row from SQLite. `search()` embeds the query with the same MiniLM model, then blends three signals per candidate: dense cosine similarity, a small chunk-type boost (review-group chunks over hotel-profile chunks), and a lexical/keyword overlap score (exact query-term hits in the chunk text or hotel name, capped at 0.15) so exact brand names and hard constraints ("parking", "pool") aren't lost to purely semantic matching. Results are deduplicated to the best-scoring chunk per hotel. `get_full_hotel_metadata()` separately looks up richer enrichment data from `data/raw/hotel_enriched_raw.json` by fuzzy name match. The pipeline and all keyword matching are English-only — there is no Turkish hint table on this path (an old one survives only in the unused `src/archive_old/retrieve.py`).

2. **Scoring** — `src/travelmind_scoring.py`: `calculate_travelmind_score()` computes a weighted 0-100 "TravelMind score" from 9 components defined in `WEIGHTS` (location_match 15, hotel_class 10, amenities_match 15, room_type_match 12, review_overall 20, review_service 8, review_rooms 8, review_cleanliness 7, review_volume 5), skipping and renormalizing around any component whose underlying data is missing rather than penalizing it. The result also carries a `missing_signals` list (human-readable names of skipped components) that `hotel_card_builder.py` copies onto each card and `ui/app.py` surfaces directly to the user as a "Score based on X/9 signals" caption — this is the whole confidence-transparency/XAI feature, so if you change the number or names of components in `WEIGHTS`, update the hardcoded `9` in `ui/app.py`'s caption to match. `build_strengths()`/`build_cautions()` derive human-readable bullet points from keyword matches in metadata/review text. `ui/app.py`'s `build_score_explanation_text()` generates its weight percentages directly from `WEIGHTS`, so that text can no longer drift out of sync with the scoring code — don't hand-write those percentages elsewhere.

3. **Card building** — `src/hotel_card_builder.py`: `build_hotel_cards()` turns raw retrieval results into structured "hotel cards" — the single source of truth the LLM and UI are both allowed to read from. It merges chunk metadata with the full enrichment record, derives YES/NO/UNKNOWN flags for amenities and room types (never guesses when data is absent), and computes a separate `rank_score` (base score adjusted by how many of the user's explicit `query_requirements` are matched/missing/unknown) used to order results.

4. **Prompting** — `src/prompt_builders.py`: builds a strict system prompt per intent (hotel search, follow-up, conversational, out-of-scope, intent-router) that forbids inventing hotels/prices/amenities, forbids exposing chain-of-thought, and forces the model to only narrate fields present in the hotel cards. `build_router_system_prompt()` defines the JSON schema (`intent`, `location`, `query_requirements`, etc.) the LLM router must return. `cmu_rag_answer.build_hotel_context()` additionally injects a deterministic "Ranking note" contrastive sentence into hotel cards ranked below the top result (naming the specific missing signals, or a generic lower-score note when nothing is missing) so the LLM can explain *why* one hotel outranks another without inventing a reason.

5. **Orchestration** — `src/cmu_rag_answer.py`: talks to Foundry Local (discovers the endpoint by shelling out to `foundry service status` and regexing the port out of its output), routes each query in two tiers — `fast_route_query()` is a zero-LLM keyword heuristic tried first (price questions, pool/breakfast follow-ups, "other hotel" follow-ups, supported/unsupported city detection); if it returns nothing, `get_llm_intent_and_location()` calls the router LLM and layers additional heuristic fallbacks/city fuzzy-matching on top of whatever JSON it returns. `generate_llm_answer()` / `generate_followup_answer()` / `generate_conversational_answer()` / `generate_out_of_scope_answer_stream()` stream the final answer. `generate_review_answer()` deliberately excludes prior chat history from the LLM request — only the current question and evidence — to stop unrelated conversation turns from bleeding into review answers.

6. **Validation** — `src/answer_validator.py`: after the LLM streams a final answer, `validate_answer()` regex/keyword-scans it for leaked reasoning, sentence-level repetition/degeneration loops, template placeholders, score-overflow, price/booking claims, unknown hotel names, unverified single-room/breakfast guarantees, and fabricated map links. Any hit replaces the whole answer with a safe templated fallback (`build_safe_fallback_answer`) rather than trying to patch the text. `scripts/evaluate_ablation.py` gives a rough quantitative read on how often this layer actually intercepts something, for demonstrating the fact-gate's effect empirically.

7. **UI** — `ui/app.py`: a single-file Streamlit app holding chat/session state (`messages`, `last_hotel_cards`, `last_search_all_cards`, `shown_hotel_count`, `selected_hotel_index`, `current_location`, `user_preferences`, `theme_internal`), custom CSS for dark/light themes, and the full intent-dispatch `if/elif` chain mirroring the intents above. `user_preferences` accumulates non-empty `query_requirements` across turns in the session (so "no pool" or "with parking" survives into a later, differently-worded search) and is merged back into `query_requirements` on every new routed query. It calls `sanitize_before_render()` as one more defensive regex strip before displaying anything, on top of `answer_validator`. Supported-city lists and city→state-name maps are duplicated across `cmu_rag_answer.py` and `ui/app.py` rather than centralized — update both if the city list changes.

8. **Data pipeline** (offline, not part of the runtime chat path) — `src/data_pipeline/`: `create_cmu_chunks.py` builds `data/processed/cmu_chunks.jsonl` from cleaned CSVs; `build_cmu_vector_db.py` / `scripts/build_embedding_matrix.py` embed those chunks into `data/cmu_travelmind.db` and the `data/embeddings.npy` matrix consumed by retrieval at runtime. The `enrich_*.py` scripts (`enrich_overpass.py`, `enrich_booking.py`, `enrich_playwright.py`, `enrich_search_rag.py`, `enrich_hotel_data.py`) populate `data/cmu_hotel_metadata.json` / `data/raw/hotel_enriched_raw.json` from external sources (OSM Overpass, Booking.com via RapidAPI, Playwright scraping, DuckDuckGo/OpenRouter/Gemini search). `src/archive_old/` and the root `archive/` directory hold superseded Phase 1-3 pipeline code and are not used by the running app.

**Security note:** `src/data_pipeline/enrich_booking.py` reads its RapidAPI key from the `RAPIDAPI_KEY` environment variable (see the module docstring/top for the exact variable name) rather than a hardcoded literal. If you ever see a real key hardcoded in that file again, treat it as already leaked (this repo is public) and get it rotated on RapidAPI's dashboard before removing it from source.
