# TravelMind RAG

TravelMind is a sophisticated, locally-hosted hotel recommendation and review-question assistant. Built with privacy and efficiency in mind, it operates entirely on your local machine using the CMU/TripAdvisor-derived dataset. By leveraging an advanced Retrieval-Augmented Generation (RAG) architecture, TravelMind provides highly accurate, fact-grounded recommendations without relying on external cloud APIs.

The application is specifically optimized to use the Qwen model (`qwen3-4b-cuda-gpu:2`) natively installed on your computer. Normal startup and preflight checks do **not** download models from the internet, ensuring a true offline-first experience.

## Core Features and Capabilities

### 1. Advanced Hybrid Retrieval (Vector + Keyword)
At the heart of TravelMind is a robust hybrid search engine. Traditional RAG systems rely solely on semantic vector search (cosine similarity), which can sometimes overlook exact-match keywords (like a specific hotel brand or a hard requirement like "parking"). To solve this, TravelMind incorporates a fast lexical (BM25-style) keyword scoring mechanism that works in tandem with the semantic search. The final retrieval score is a weighted combination of semantic similarity, hotel type boosting, and keyword matching, ensuring you get the most accurate and relevant results every time.

### 2. Confidence Transparency & Explainable AI (XAI)
We believe that AI decision-making should not be a black box. TravelMind's scoring algorithm evaluates hotels across 9 distinct data signals (such as hotel class, service rating, cleanliness, etc.). When data is missing for a specific hotel, the system explicitly communicates this to the user. The UI displays exactly how many signals were used to calculate the final score and lists which specific metrics were missing (e.g., *"Score based on 7/9 signals. Missing: Hotel class, Parking"*). This provides a transparent, trustworthy view of the underlying data structure.

### 3. Session Preference Accumulation (Contextual Memory)
Real conversations flow naturally, and an AI assistant should keep up. TravelMind's Streamlit-based UI features intelligent preference accumulation. If you request a "hotel with a pool" and later ask "What about in Dallas?", the system remembers your previous requirement for a pool. It seamlessly merges your intent across conversational turns, providing a fluid and context-aware user experience without requiring you to repeat your preferences.

### 4. Contrastive Explanation for Rankings
When presenting multiple options, users often wonder why one hotel is ranked lower than another. TravelMind solves this by dynamically injecting deterministic contrastive notes into the LLM's context window. By comparing the missing signals or score differences between the top results, the model can naturally explain to the user why a specific hotel was placed in the second or third spot (e.g., *"Ranked below the first option due to missing parking data"*).

### 5. Robust Fact-Gate Validator (Ablation Tested)
To strictly prevent hallucinations—such as inventing fake amenities, imaginary prices, or broken links—TravelMind employs a rigorous deterministic Fact-Gate. After the LLM generates an answer, the validator intercepts it and checks for any unverified claims. If a hallucination is detected, the system safely falls back to a grounded, template-based response. We have included an ablation testing suite (`scripts/evaluate_ablation.py`) to quantitatively measure and prove the hallucination-reduction impact of this validator.

### 6. Review-Based Follow-Up Questions
Beyond just recommending hotels, TravelMind can answer specific follow-up questions by searching through actual guest reviews. Want to know if the Wi-Fi is reliable, or if the rooms are noisy? The system retrieves the most relevant review chunks for the currently selected hotel and generates a grounded response based purely on real guest experiences.

## Local Architecture

- **Retrieval:** 
  The lightweight `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` model runs purely on the CPU. It searches the normalized local embedding matrix using cosine similarity, keeping your GPU memory completely free for the generative model.
- **Scoring & Ranking:** 
  The 100-point ranking system is defined in `src/travelmind_scoring.py`. It integrates numeric review averages and logarithmically normalized review volumes from `data/hotel_review_stats.json`. Missing ratings are intelligently skipped and the remaining weights are renormalized.
- **Generation:** 
  Text generation is powered by `qwen3-4b-cuda-gpu:2` running via Foundry Local on the RTX 3050. We optimize for speed and safety by disabling continuous "thinking" for user-facing outputs and enforcing strict `<answer>` tag boundaries before extraction and validation.
- **Review RAG:** 
  Review questions search only within the specific review chunks belonging to the selected hotel. The AI is strictly instructed to use the supplied evidence and is blocked from inventing prices, reservation states, or external links.

## Required Local Artifacts

The large runtime artifacts are ignored by Git (as configured in `.gitignore`). A clean clone is therefore not runnable until these existing files and caches have been copied into place:

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

*Note: `data/raw/cmu_tripadvisor/review.txt` is required only when rebuilding review statistics, not for normal chat operations.*

## Install and Run on Windows

Python 3.12 is the tested and recommended version. From the repository root, run:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

First, run the read-only preflight check. It verifies every required data file, ensures exact database/embedding ID alignment, checks both local model caches, and never starts or downloads a model:

```powershell
.venv\Scripts\python.exe scripts\verify_runtime_artifacts.py
```

Then, launch the Streamlit User Interface:

```powershell
.venv\Scripts\python.exe -m streamlit run ui\app.py
```

TravelMind automatically starts the installed Foundry Local service when needed and loads the already-cached Qwen model into VRAM. Do not run `scripts\setup_foundry_runtime.py` merely to start the app; that script is only for explicitly preparing a new machine. Its default mode reuses cached artifacts only; internet downloads require the separate, explicit `--allow-download` flag.

## Rebuild Derived Data

After modifying the raw reviews, update the stats:

```powershell
.venv\Scripts\python.exe scripts\build_review_stats.py
```

After modifying the database `chunks` table, rebuild the embeddings and indexes:

```powershell
.venv\Scripts\python.exe scripts\build_embedding_matrix.py
.venv\Scripts\python.exe scripts\build_retrieval_row_index.py
.venv\Scripts\python.exe scripts\verify_runtime_artifacts.py
```

## Regression Checks & Testing

To ensure the integrity of the system after making modifications, run the full test suite:

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts\test_review_questions.py
.venv\Scripts\python.exe scripts\benchmark_detroit.py
```

- **The Detroit benchmark** performs five foreground runs and reports first-token latency, total generation time, answer length, preamble extraction success, and validator fallback rates.
- **The review script** exercises the required rooms/noise/service questions and prints the selected hotel, evidence count, validator messages, and fallback state.
