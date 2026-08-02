# TravelMind RAG - AI Travel Assistant 🌍🏨

Welcome to **TravelMind**, a highly advanced, locally-hosted Retrieval-Augmented Generation (RAG) system designed to act as an AI travel and hotel recommendation assistant. 

## 📖 Project History & Evolution

Our journey in building TravelMind has gone through several critical phases, evolving from a simple data parsing script into a fully autonomous, hallucination-free AI assistant.

### Phase 1: Data Acquisition & Preprocessing
- **Kaggle Dataset:** We started by processing a massive hotel dataset from Kaggle.
- **Parsing & Cleaning:** The initial codebase focused heavily on data cleaning. We extracted unstructured text, metadata, reviews, and amenities, mapping them into structured dictionaries using Python.
- **Chunking Strategy:** We implemented an overlapping chunking algorithm to split long hotel descriptions and reviews into semantic blocks, ensuring that the AI context window wouldn't overflow while maintaining context.

### Phase 2: Initial Vectorization & Search (The Legacy Architecture)
- **Sentence-Transformers:** Initially, we used the `sentence-transformers` library to convert our text chunks into embeddings.
- **Numpy Vector Store:** The semantic vectors were saved as local `.npy` files for cosine similarity calculations.
- **Legacy Database (`travelmind_old_kaggle.db`):** All metadata and chunks were stored in a massive SQLite database. *(Note: Due to GitHub file limits, this legacy database is now split and archived in this repo).*

### Phase 3: The Hybrid Local RAG Architecture (Current)
- **Massive Scale Engineering:** The initial project plan assumed a small, experimental dataset (5-10 documents). Because TravelMind operates on a massive real-world dataset (over **224,500** chunks), generating embeddings via an LLM endpoint would take hours. Therefore, we made the architectural decision to retain the highly-optimized **PyTorch & Sentence-Transformers (`MiniLM`)** pipeline for ultra-fast embedding generation and vector retrieval.
- **Foundry Local for LLM Chat:** To eliminate API latency, `APIConnectionError` issues, and dependency on external servers, we integrated the **Foundry Local SDK** exclusively for the Generation (Chat) phase. The local LLM processes the retrieved context offline, ensuring 100% data privacy and zero network calls.
- **New SQLite Database:** We rebuilt the database (`cmu_travelmind.db`) to store JSON-serialized metadata and vector embeddings, massively improving read/write speeds and structural integrity compared to the legacy `.npy` files.

### Phase 4: Prompt Engineering & Guardrails
- **Zero Hallucination Policy:** We engineered strict LLM system prompts that force the AI to *only* answer using the provided database context. 
- **Internal Reasoning (`<think>`):** The model was configured to use a `<think>` block before answering, analyzing the data internally before presenting a refined output to the user.

### Phase 5: Multi-Lingual Streamlit User Interface
- We built a fully responsive web interface using **Streamlit**.
- **Dynamic Theming:** Custom CSS ensures perfect compatibility with both Dark and Light modes, solving background clipping and text readability issues.
- **Global Reach:** Added real-time UI translation and NLP support for English, Turkish, German, French, Italian, and Chinese. 

---

## ✨ Core Features
- **Hybrid Local AI Models:** Blazing-fast PyTorch embeddings combined with self-hosted Foundry Local LLM inference, ensuring 100% data privacy.
- **Multi-language Support:** UI and NLP processing natively supported in 6 languages.
- **Theme-responsive UI:** Streamlit frontend adapts to light and dark themes seamlessly.
- **Hallucination Guardrails:** Strict prompt engineering ensures the AI never guesses outside of the provided context.

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/bilalkaplan/TravelMind.git
   cd TravelMind
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Local AI Engine:**
   Ensure the Foundry Local Service is running in the background.
   ```bash
   foundry service start
   ```

4. **Launch the Web Interface:**
   ```bash
   streamlit run ui/app.py
   ```

---

## 📦 Note on the Legacy Database Archive
Due to GitHub's 100MB file size limit, the legacy database (`archive_old/old_databases/travelmind_old_kaggle.db`, ~245MB) from Phase 2 was compressed into a zip archive and then split into two parts:
- `travelmind_old_kaggle.zip.part1`
- `travelmind_old_kaggle.zip.part2`

If you need to access this old database for historical testing, you can merge the parts and extract them:
```bash
# On Windows PowerShell
cat archive_old/old_databases/travelmind_old_kaggle.zip.part* > archive_old/old_databases/travelmind_old_kaggle.zip
```
