# TravelMind RAG: An Offline-First, Privacy-Preserving Hotel Assistant

Welcome to the TravelMind repository. This project represents a fully local, privacy-first AI assistant engineered specifically for hotel recommendation and review-based question answering. My primary motivation for building this system was to develop a sophisticated Retrieval-Augmented Generation (RAG) architecture that operates entirely offline, completely eliminating any reliance on external cloud APIs or third-party data processors. 

By leveraging the comprehensive CMU/TripAdvisor dataset, TravelMind is capable of reasoning over real-world hotel data, semantic reviews, and complex user constraints directly on consumer hardware.

---

## Architectural Philosophy and Advanced Features

Standard RAG pipelines often suffer from hallucination, context drift, and an inability to strictly adhere to hard constraints. To overcome these limitations, I engineered several advanced mechanisms into the core architecture of TravelMind.

### 1. Hybrid Retrieval Engine (Semantic + Lexical)
Relying solely on dense vector retrieval (cosine similarity) frequently results in missed exact-match keywords—particularly for specific brand names (e.g., "Marriott") or hard boolean constraints (e.g., "parking", "pool"). To address this, I implemented a fast, BM25-inspired lexical keyword scoring system that operates in tandem with the semantic search. The final retrieval score is a weighted ensemble of semantic similarity, categorical boosting, and lexical overlap, ensuring high-precision recall for exact user constraints.

### 2. Transparent Explainability (XAI) and Deterministic Scoring
I strongly believe that AI systems should not operate as black boxes, especially in recommendation contexts. TravelMind evaluates candidates across 9 distinct, individually-weighted deterministic signals (location match, hotel class, amenities match, room-type match, and five review-derived ratings covering overall, service, rooms, cleanliness, and review volume). If specific data points are missing for a candidate, the UI explicitly flags this gap. You will see transparent logs such as *"Score based on 7/9 signals. Missing: Hotel class, Parking"*. This guarantees that the user is always aware of the exact underlying data driving the recommendation.

### 3. Contextual Session Memory (Multi-Turn Accumulation)
Natural human conversations are highly contextual. I integrated a robust session state memory mechanism that seamlessly tracks constraints across multi-turn interactions. If a user initially requests a "hotel with a pool" and subsequently asks, "What about in Dallas?", the system dynamically merges these intents. It carries forward previous constraints without requiring the user to repetitively state their requirements.

### 4. Dynamic Contrastive Explanations
When the system ranks one candidate below another, it dynamically injects contrastive reasoning notes into the Large Language Model's (LLM) context window. The model synthesizes these data points to provide natural, logical explanations for the ranking hierarchy (e.g., *"This option is ranked slightly lower due to missing parking data and a marginally lower cleanliness rating"*).

### 5. Strict Fact-Gate Validator (Zero-Hallucination Enforcement)
To strictly enforce factual accuracy and prevent the generative model from inventing non-existent amenities or hallucinating live prices, I developed a deterministic Fact-Gate Validator. This post-processing layer intercepts the LLM's raw output and rigorously cross-references any claims made against the retrieved ground-truth data. If an unsupported claim is detected, the system safely falls back to a grounded, template-based response. An ablation testing script (`scripts/evaluate_ablation.py`) is provided for empirical validation of this safety mechanism.

---

## Hardware Optimization and Technical Stack

Operating a complex RAG system within a strict 4GB VRAM constraint required aggressive resource management and workload distribution:

*   **Retrieval (CPU Bound):** The dense embedding model (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) runs entirely on the CPU. The system queries a pre-computed, normalized local embedding matrix using highly optimized NumPy operations, ensuring zero VRAM overhead during the retrieval phase.
*   **Deterministic Scoring Engine:** The internal 100-point ranking algorithm (`src/travelmind_scoring.py`) intelligently balances numeric review averages and log-normalized review volumes. It handles missing data gracefully by recalculating weights dynamically, ensuring a fair and mathematically sound comparison across diverse candidates.
*   **Generative AI (GPU Bound):** Text generation is handled by the `qwen3-4b` model alias, resolved and served natively via Foundry Local on an RTX 3050. To maintain low latency and prevent reasoning leaks, continuous "thinking" is disabled for user-facing outputs, and strict XML-style `<answer>` tags are enforced and parsed.
*   **Review-Specific RAG Pipeline:** When answering follow-up queries regarding specific hotels, the retrieval space is dynamically restricted to review chunks belonging exclusively to the target hotel. The LLM is strictly prompted to synthesize answers based solely on this constrained evidence pool.

---

## Required Local Data Artifacts

Due to standard version control size constraints, the large runtime databases and pre-computed embedding matrices are excluded via `.gitignore`. To run the system locally, the following artifacts must be present in the `data/` directory:

| Artifact | Description |
| :--- | :--- |
| `data/cmu_travelmind.db` | The primary SQLite database containing the structured hotel data and raw reviews. |
| `data/embeddings.npy` | Pre-computed, normalized L2 chunk embeddings utilized for high-speed semantic retrieval. |
| `data/embedding_ids.npy` | Index mapping array linking matrix rows to exact database identifiers. |
| `data/retrieval_row_index.npz` | Compressed cache containing fast startup indices and per-hotel review associations. |
| `data/hotel_review_stats.json` | Pre-calculated statistical aggregates for hotel reviews. |
| `data/cmu_hotel_metadata.json` | Core structural metadata including baseline hotel names and geospatial locations. |
| `data/raw/hotel_enriched_raw.json` | Extracted auxiliary data, including specific amenities and recorded room type configurations. |

*Note: Execution requires the local Hugging Face cache (for the MiniLM transformer) and the Foundry Local model cache (for the Qwen LLM) to be present on your workstation.*

---

## Installation and Execution (Windows Environment)

The system was developed and rigorously tested under Python 3.12.

**1. Virtual Environment Initialization:**
```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**2. Artifact Verification:**
Prior to launching the application, execute the preflight diagnostic check. This script verifies the integrity and alignment of all local data files without initiating any network requests:
```powershell
.venv\Scripts\python.exe scripts\verify_runtime_artifacts.py
```

**3. Launching the Application:**
Initialize the Streamlit user interface. The system will automatically interface with Foundry Local and load the required generative model into VRAM.
```powershell
.venv\Scripts\python.exe -m streamlit run ui\app.py
```

---

## Data Pipeline Management and Testing

Should you need to update or modify the core CMU dataset, the following pipeline scripts must be executed sequentially to rebuild the statistical aggregates and embedding matrices:

```powershell
.venv\Scripts\python.exe scripts\build_review_stats.py
.venv\Scripts\python.exe scripts\build_embedding_matrix.py
.venv\Scripts\python.exe scripts\build_retrieval_row_index.py
.venv\Scripts\python.exe scripts\verify_runtime_artifacts.py
```

### Comprehensive Testing Suite

To ensure absolute system stability during continuous development, I have included a comprehensive suite of automated tests. It is highly recommended to run these after any architectural modifications:

```powershell
# Execute the primary unit and integration test suite
.venv\Scripts\python.exe -m pytest -q

# Validate review extraction logic and fallback mechanism integrity
.venv\Scripts\python.exe scripts\test_review_questions.py

# Execute the Detroit performance benchmark (measures latency, generation throughput, and fallback frequency)
.venv\Scripts\python.exe scripts\benchmark_detroit.py
```
