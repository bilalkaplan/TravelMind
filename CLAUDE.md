# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

TravelMind is a locally-hosted RAG hotel-recommendation assistant (see `project_scope.md` and `README.md` for full history). It only answers hotel/accommodation questions for a fixed set of ~25 US cities, using a CMU/TripAdvisor-derived dataset of hotels and reviews. It intentionally does not do flights, visas, itineraries, or live pricing/booking.

Two local models power it, both chosen to avoid network calls and API costs:
- **Embeddings**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (PyTorch, local) for retrieval over ~224k chunks.
- **Generation**: a local LLM served by **Foundry Local** (`foundry service start`), accessed through the `openai` SDK pointed at the discovered local endpoint — no API key, no external network calls at chat time.

## Commands

Python env lives in `.venv/` (Windows). Use it directly rather than relying on `requirements.txt`, which is UTF-16 encoded and out of date relative to what's actually installed (streamlit, openai, playwright, pytest, etc. are present in `.venv` but missing from the file). If you need to add a dependency, `pip install` it into `.venv` and then regenerate the file with `pip freeze` (as UTF-8) rather than hand-editing it.

```powershell
# Run the full backend test suite
.venv\Scripts\python.exe -m pytest tests\ -q

# Run a single test file / test
.venv\Scripts\python.exe -m pytest tests\test_scoring.py -q
.venv\Scripts\python.exe -m pytest tests\test_router.py::test_router_dirty_json -q

# Start the local LLM service (separate terminal, required before chatting)
foundry service start

# Launch the Streamlit UI
.venv\Scripts\streamlit run ui\app.py

# Backend-only smoke test over tests\test_suite.json (no browser), logs to ui_test_logs\test_results.md
.venv\Scripts\python.exe run_backend_tests.py

# Full UI test via Playwright (needs the Streamlit server already running on :8501)
.venv\Scripts\python.exe run_ui_tests.py
```

`tests/test_parser.py` currently fails to collect — it imports `stream_and_strip_think` from `src/cmu_rag_answer.py`, which no longer exists (streaming no longer separates `<think>` tags into that function; see `stream_extract_answer` instead). Running `pytest tests/` as a whole will error out at collection; use `--ignore=tests/test_parser.py` or target files individually until that test is updated or removed.

`src/` is not a package (no `__init__.py`); every entry point (`ui/app.py`, `tests/conftest.py`, `run_backend_tests.py`) manually does `sys.path.insert/append` to make `src` modules importable. New modules under `src/` should follow the existing flat-import style (`from cmu_retrieve import search`, not package-relative imports).

## Architecture

The core design is a **layered anti-hallucination pipeline** — the LLM is never trusted to know or invent hotel facts; it only narrates data that was already verified in code:

1. **Retrieval** — `src/cmu_retrieve.py`: loads all chunks + their precomputed embeddings from the SQLite DB (`data/cmu_travelmind.db`, table `chunks`, columns include `embedding_json`/`metadata_json`), embeds the query with the same MiniLM model, does dot-product scoring with a Turkish→English keyword-expansion table (`TURKISH_TO_ENGLISH_HINTS`) so Turkish queries still match English review text, applies a small type boost for review-group chunks over hotel-profile chunks, then deduplicates to the best-scoring chunk per hotel. `get_full_hotel_metadata()` separately looks up richer enrichment data from `data/raw/hotel_enriched_raw.json` by fuzzy name match.

2. **Scoring** — `src/travelmind_scoring.py`: `calculate_travelmind_score()` computes a weighted 0-100 "TravelMind score" per candidate (location match, hotel class, amenities match, room-type match, review-count trust signal, cleanliness sentiment, phone/map-data presence), skipping any component whose underlying data is missing rather than penalizing it. `build_strengths()`/`build_cautions()` derive human-readable bullet points from keyword matches in metadata/review text. Note: the weights actually used here (location 20 / class 25 / amenities 25 / room 25 / review 5 / cleanliness 5 / phone 2 / map 3) do **not** match the percentages hardcoded into the `score_explanation` reply text in `ui/app.py` (35/25/15/15/5/5) — if you change one, check the other.

3. **Card building** — `src/hotel_card_builder.py`: `build_hotel_cards()` turns raw retrieval results into structured "hotel cards" — the single source of truth the LLM and UI are both allowed to read from. It merges chunk metadata with the full enrichment record, derives YES/NO/UNKNOWN flags for amenities and room types (never guesses when data is absent), and computes a separate `rank_score` (base score adjusted by how many of the user's explicit `query_requirements` are matched/missing/unknown) used to order results.

4. **Prompting** — `src/prompt_builders.py`: builds a strict system prompt per intent (hotel search, follow-up, conversational, out-of-scope, intent-router) that forbids inventing hotels/prices/amenities, forbids exposing chain-of-thought, and forces the model to only narrate fields present in the hotel cards. `build_router_system_prompt()` defines the JSON schema (`intent`, `location`, `query_requirements`, etc.) the LLM router must return.

5. **Orchestration** — `src/cmu_rag_answer.py`: talks to Foundry Local (discovers the endpoint by shelling out to `foundry service status` and regexing the port out of its output), routes each query in two tiers — `fast_route_query()` is a zero-LLM keyword heuristic tried first (price questions, pool/breakfast follow-ups, "other hotel" follow-ups, supported/unsupported city detection); if it returns nothing, `get_llm_intent_and_location()` calls the router LLM and layers additional heuristic fallbacks/city fuzzy-matching on top of whatever JSON it returns. `generate_llm_answer()` / `generate_followup_answer()` / `generate_conversational_answer()` stream the final answer.

6. **Validation** — `src/answer_validator.py`: after the LLM streams a final answer, `validate_answer()` regex/keyword-scans it for leaked reasoning, template placeholders, score-overflow, price/booking claims, unknown hotel names, unverified single-room/breakfast guarantees, and fabricated map links. Any hit replaces the whole answer with a safe templated fallback (`build_safe_fallback_answer`) rather than trying to patch the text.

7. **UI** — `ui/app.py`: a single-file Streamlit app holding chat/session state (`messages`, `last_hotel_cards`, `selected_hotel_index`, `current_location`, `language`, `theme_internal`), custom CSS for dark/light themes, and the full intent-dispatch `if/elif` chain mirroring the intents above. It calls `sanitize_before_render()` as one more defensive regex strip before displaying anything, on top of `answer_validator`. Supported-city lists and city→state-name maps are duplicated across `cmu_rag_answer.py` and `ui/app.py` rather than centralized — update both if the city list changes.

8. **Data pipeline** (offline, not part of the runtime chat path) — `src/data_pipeline/`: `create_cmu_chunks.py` builds `data/processed/cmu_chunks.jsonl` from cleaned CSVs; `build_cmu_vector_db.py` embeds those chunks into `data/cmu_travelmind.db`. The `enrich_*.py` scripts (`enrich_overpass.py`, `enrich_booking.py`, `enrich_playwright.py`, `enrich_search_rag.py`, `enrich_hotel_data.py`) populate `data/cmu_hotel_metadata.json` / `data/raw/hotel_enriched_raw.json` from external sources (OSM Overpass, Booking.com via RapidAPI, Playwright scraping, DuckDuckGo search). `src/archive_old/` and the root `archive/` directory hold superseded Phase 1-3 pipeline code and are not used by the running app.

`src/language_utils.py` provides rule-based (character/keyword) Turkish/English detection and static bilingual message strings; the live chat path in `ui/app.py` actually drives language off the UI's language selector (`t["code"]`), not this detector. `src/translation_gateway.py` is an unused pass-through stub for a future translation provider — `translate_to_english`/`translate_from_english` currently just return the input unchanged.

**Security note:** `src/data_pipeline/enrich_booking.py` has a RapidAPI key hardcoded at module level (`API_KEY = "..."`). It's a data-ingestion script not part of the runtime app, but treat that key as already exposed if you touch this file.
