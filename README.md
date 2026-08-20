# TravelMind RAG: My Offline-First Hotel Assistant

Hey there! 👋 Welcome to TravelMind. This is a privacy-first, fully local AI assistant I built to recommend hotels and answer specific review-based questions. I wanted a system that runs 100% offline using the CMU/TripAdvisor dataset, so it doesn't rely on any external cloud APIs. 

I've optimized this to run smoothly on consumer hardware. It uses the `qwen3-4b-cuda-gpu:2` model natively via Foundry Local. Normal startup doesn't download any models from the internet—it's a true offline-first experience.

---

## 🌟 Advanced RAG Features I've Added

While building this, I realized standard RAG pipelines have some limitations, so I engineered a few advanced enhancements to make the system much more reliable and prevent hallucinations.

### 1. Hybrid Retrieval (Vector + Lexical)
I noticed that pure semantic vector search (cosine similarity) sometimes misses exact-match keywords (like a specific brand such as "Marriott" or a hard requirement like "parking"). To fix this, I added a fast lexical (BM25-style) keyword scoring mechanism that runs alongside the semantic search. The final score is a weighted mix of semantic similarity, hotel type boosting, and keyword overlap. It works much better for exact user constraints!

### 2. Confidence Transparency & Explainable AI (XAI)
I really don't like AI black boxes. So, I made sure TravelMind evaluates hotels across 9 distinct deterministic signals (like hotel class, service rating, cleanliness). If data is missing, the UI explicitly tells you. You'll see messages like *"Score based on 7/9 signals. Missing: Hotel class, Parking"*. This way, you always know exactly what data the recommendation is based on.

### 3. Session Preference Accumulation (Contextual Memory)
Real conversations flow, so I added a contextual memory feature to the Streamlit session state. If you ask for a "hotel with a pool" and then say, "What about in Dallas?", the system remembers that you still want a pool. It merges intents across turns naturally, so you don't have to keep repeating yourself.

### 4. Contrastive Explanation for Rankings
Whenever the system ranks one hotel below another, users naturally wonder *why*. I solved this by dynamically injecting contrastive notes into the LLM's context. The model compares missing signals or score differences and naturally explains the ranking to you (e.g., *"Ranked below the first option because it's missing parking data"*).

### 5. Robust Fact-Gate Validator
To make sure the AI *never* hallucinates (like inventing fake amenities or prices), I built a strict deterministic Fact-Gate. After the LLM generates an answer, this validator intercepts it and checks for unverified claims against the actual retrieved data. If it catches a hallucination, it safely falls back to a grounded template. I've even included an ablation testing script (`scripts/evaluate_ablation.py`) if you want to test how effective this validator is.

---

## 🏗️ Technical Architecture

Since I had to make this work well on a 4GB VRAM limit, I split the workloads between the CPU and GPU:

- **Retrieval (CPU):** I put the lightweight `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` model on the CPU. It searches the normalized local embedding matrix using fast NumPy operations, leaving the GPU memory completely free.
- **Deterministic Scoring Engine:** The 100-point ranking system (`src/travelmind_scoring.py`) combines numeric review averages and log-normalized review volumes. I made sure it skips missing ratings and renormalizes weights for a fair comparison.
- **Generative AI (GPU):** Text generation runs on `qwen3-4b-cuda-gpu:2` via Foundry Local on the RTX 3050. To keep things fast and safe, I disabled continuous "thinking" for user-facing outputs and enforced strict `<answer>` tags before validation.
- **Review RAG Pipeline:** For follow-up questions, the system filters the search space to *only* include review chunks for that specific hotel. The AI is strictly prompted to only use the supplied evidence.

---

## 📦 Required Local Artifacts

Because of GitHub's size limits, the large runtime databases and embedding matrices are ignored in `.gitignore`. If you're cloning this, it won't run until you place these files in the `data/` directory:

| Artifact | Purpose |
| --- | --- |
| `data/cmu_travelmind.db` | The main SQLite database with hotels and raw reviews. |
| `data/embeddings.npy` | Pre-computed, normalized chunk embeddings for fast retrieval. |
| `data/embedding_ids.npy` | Maps matrix indices to exact database rows. |
| `data/retrieval_row_index.npz` | Fast startup caching and per-hotel review memberships. |
| `data/hotel_review_stats.json` | Pre-calculated per-hotel review aggregates. |
| `data/cmu_hotel_metadata.json` | Base hotel metadata (names, locations). |
| `data/raw/hotel_enriched_raw.json` | Scraped rich amenities and recorded room types. |

*Note: You'll also need the local Hugging Face cache (for MiniLM) and the Foundry Local cache (for Qwen) on your machine.*

---

## 🚀 Setup & Run (Windows)

I tested and built this using Python 3.12. 

**1. Setup the Virtual Environment:**
```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**2. Verify Artifacts:**
Before launching, run this preflight check. It verifies all data files and alignment without triggering any downloads:
```powershell
.venv\Scripts\python.exe scripts\verify_runtime_artifacts.py
```

**3. Launch the App:**
Start the Streamlit UI. It will automatically start Foundry Local and load the Qwen model.
```powershell
.venv\Scripts\python.exe -m streamlit run ui\app.py
```

---

## 🛠️ Data Pipeline & Tests

If you modify the core dataset, you'll need to rebuild the stats and matrices:

```powershell
.venv\Scripts\python.exe scripts\build_review_stats.py
.venv\Scripts\python.exe scripts\build_embedding_matrix.py
.venv\Scripts\python.exe scripts\build_retrieval_row_index.py
.venv\Scripts\python.exe scripts\verify_runtime_artifacts.py
```

I also wrote a suite of automated tests to make sure everything stays intact when making changes:

```powershell
# Run the full test suite
.venv\Scripts\python.exe -m pytest -q

# Test the review question extraction and fallback states
.venv\Scripts\python.exe scripts\test_review_questions.py

# Run my Detroit performance benchmark (latency, generation time, fallback rates)
.venv\Scripts\python.exe scripts\benchmark_detroit.py
```
