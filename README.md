# TravelMind RAG - AI Travel Assistant

## Overview
TravelMind is an AI-powered travel assistant that uses a Retrieval-Augmented Generation (RAG) architecture to recommend hotels. It leverages local embedding models (Qwen3 via Foundry Local SDK) and vector search to find the perfect hotel based on user criteria.

## Features
- **Local AI Models:** Completely self-hosted vector embeddings and LLM inference ensuring data privacy.
- **Multi-language Support:** UI and NLP processing available in English, Turkish, German, French, Italian, and Chinese.
- **Theme-responsive UI:** Streamlit frontend adapts to light and dark themes seamlessly.
- **Hallucination Guardrails:** Strict prompt engineering ensures the AI never guesses outside of the provided database context.

## Installation
1. Clone the repository.
2. Run `pip install -r requirements.txt`.
3. Ensure the Foundry Local Service is running (`foundry service start`).
4. Run the app: `streamlit run ui/app.py`

## Note on Old Database
Due to GitHub's file size limits, the legacy database (`archive_old/old_databases/travelmind_old_kaggle.db`, ~245MB) was compressed into a zip archive and then split into two parts (`travelmind_old_kaggle.zip.part1` and `travelmind_old_kaggle.zip.part2`). If you need to access this old database, you can merge the parts and extract them:
```bash
# On Windows PowerShell
cat travelmind_old_kaggle.zip.part* > travelmind_old_kaggle.zip
```
