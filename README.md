# TravelMind RAG: An Offline-First, High-Precision Hotel Assistant

TravelMind is a sophisticated, privacy-first, locally-hosted AI assistant designed to recommend hotels and answer detailed review-based follow-up questions. Built entirely on the CMU/TripAdvisor-derived dataset, TravelMind operates 100% offline. It leverages an advanced Retrieval-Augmented Generation (RAG) architecture to provide highly accurate, fact-grounded recommendations without relying on external cloud APIs or risking user privacy.

The application is heavily optimized to run on consumer hardware, specifically utilizing the `qwen3-4b-cuda-gpu:2` model natively via Foundry Local. Normal startup and preflight checks do **not** download models from the internet, guaranteeing a true offline-first experience.

---

## 🌟 Advanced RAG Innovations

We have engineered several academic-grade enhancements to push the boundaries of standard RAG pipelines, ensuring absolute reliability and zero hallucinations.

### 1. Hybrid Retrieval (Vector + Lexical)
Traditional RAG systems rely solely on semantic vector search (cosine similarity), which often overlooks exact-match keywords (e.g., a specific hotel brand like "Marriott" or a hard requirement like "parking"). TravelMind incorporates a fast lexical (BM25-style) keyword scoring mechanism running in tandem with the semantic search. The final retrieval score is a dynamic weighted combination of semantic similarity, hotel type boosting, and keyword overlap, ensuring absolute precision for exact user constraints.

### 2. Confidence Transparency & Explainable AI (XAI)
AI decision-making should not be a black box. TravelMind evaluates hotels across 9 distinct deterministic data signals (e.g., hotel class, service rating, cleanliness, value). When data is missing for a specific hotel, the system explicitly communicates this. The Streamlit UI displays exactly how many signals were used and lists the missing metrics (e.g., *"Score based on 7/9 signals. Missing: Hotel class, Parking"*). This fulfills strict Explainable AI (XAI) standards.

### 3. Session Preference Accumulation (Contextual Memory)
Conversations flow naturally, and TravelMind keeps up. The intelligent Streamlit session state accumulates user preferences across multiple conversational turns. If a user requests a "hotel with a pool" and later asks, "What about in Dallas?", the system remembers the pool requirement. It seamlessly merges intents across turns, providing a fluid, context-aware experience without forcing the user to repeat constraints.

### 4. Contrastive Explanation for Rankings
When presenting multiple hotel options, users intuitively wonder why one ranked lower than another. TravelMind dynamically injects deterministic contrastive notes into the LLM's context window. By comparing the missing signals or score differences between the top results, the model can naturally explain the ranking logic to the user (e.g., *"Ranked below the first option due to missing parking data"*).

### 5. Robust Fact-Gate Validator (Ablation Tested)
To guarantee zero hallucinations (inventing fake amenities, imaginary prices, or broken links), TravelMind employs a rigorous deterministic Fact-Gate. After the LLM generates an answer, the validator intercepts the stream and checks for unverified claims against the retrieved metadata. If a hallucination is detected, the system safely falls back to a grounded, template-based response. The effectiveness of this module is quantitatively proven via our custom ablation testing suite (`scripts/evaluate_ablation.py`).

---

## 🏗️ Technical Architecture

TravelMind intelligently partitions workloads between the CPU and GPU to maximize performance on constrained hardware (e.g., 4GB VRAM limit).

- **Embedding & Retrieval (CPU):** 
  The lightweight `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` model runs purely on the CPU. It searches the normalized local embedding matrix using highly optimized NumPy operations, keeping GPU memory completely free for the generative model.
- **Deterministic Scoring Engine:** 
  The 100-point ranking system (`src/travelmind_scoring.py`) integrates numeric review averages and logarithmically normalized review volumes from `data/hotel_review_stats.json`. Missing ratings are intelligently skipped, and the remaining weights are dynamically renormalized to ensure fair comparisons.
- **Generative AI (GPU):** 
  Text generation is powered by `qwen3-4b-cuda-gpu:2` running via Foundry Local on an RTX 3050. We optimize for speed and safety by disabling continuous "thinking" for user-facing outputs and enforcing strict `<answer>` tag boundaries before extraction and validation.
- **Review RAG Pipeline:** 
  For follow-up questions (e.g., *"Is the room noisy?"*), the system filters the search space to *only* include review chunks belonging to the selected hotel. The AI is strictly instructed to answer using only the supplied evidence.

---

## 📦 Required Local Artifacts

Due to size constraints, large runtime databases and embedding matrices are ignored by Git (as configured in `.gitignore`). A clean clone is not runnable until these required files are placed in the `data/` directory:

| Artifact | Purpose |
| --- | --- |
| `data/cmu_travelmind.db` | Primary SQLite database containing hotels and raw review chunks. |
| `data/embeddings.npy` | Pre-computed, normalized chunk embeddings for lightning-fast retrieval. |
| `data/embedding_ids.npy` | Exact database-row alignment to map matrix indices to database rows. |
| `data/retrieval_row_index.npz` | Fast startup/filter caching and per-hotel review memberships. |
| `data/hotel_review_stats.json` | Pre-calculated per-hotel numeric review aggregates (averages/volumes). |
| `data/cmu_hotel_metadata.json` | Base hotel metadata (names, locations, core info). |
| `data/raw/hotel_enriched_raw.json` | Scraped rich amenities and recorded room types. |

*Note: The local Hugging Face cache (for MiniLM) and the Foundry Local cache (for Qwen) must also be present on the host machine.*

---

## 🚀 Install and Run (Windows)

Python 3.12 is the tested and recommended version. 

**1. Setup the Virtual Environment:**
```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**2. Verify Runtime Artifacts:**
Before launching the app, run the read-only preflight check. It verifies every required data file, ensures exact database/embedding ID alignment, and checks local model caches without triggering any downloads:
```powershell
.venv\Scripts\python.exe scripts\verify_runtime_artifacts.py
```

**3. Launch the Application:**
Start the Streamlit User Interface. TravelMind will automatically start the installed Foundry Local service when needed and load the already-cached Qwen model into VRAM.
```powershell
.venv\Scripts\python.exe -m streamlit run ui\app.py
```

---

## 🛠️ Data Pipeline & Maintenance

If you update the core dataset, you must rebuild the derived statistics and embedding matrices.

**Rebuild Review Statistics:**
```powershell
.venv\Scripts\python.exe scripts\build_review_stats.py
```

**Rebuild Embeddings and Indexes:**
```powershell
.venv\Scripts\python.exe scripts\build_embedding_matrix.py
.venv\Scripts\python.exe scripts\build_retrieval_row_index.py
.venv\Scripts\python.exe scripts\verify_runtime_artifacts.py
```

---

## 🧪 Regression Testing & Benchmarking

TravelMind includes a rigorous suite of automated tests to ensure system integrity after modifications.

**Run the Full Test Suite:**
```powershell
.venv\Scripts\python.exe -m pytest -q
```

**Evaluate Review Question Extraction:**
Exercises the required rooms/noise/service questions and prints the selected hotel, evidence count, validator messages, and fallback state.
```powershell
.venv\Scripts\python.exe scripts\test_review_questions.py
```

**Detroit Performance Benchmark:**
Performs five consecutive foreground runs simulating searches for "Detroit". It reports critical metrics including first-token latency, total generation time, answer length, preamble extraction success, and validator fallback rates.
```powershell
.venv\Scripts\python.exe scripts\benchmark_detroit.py
```
