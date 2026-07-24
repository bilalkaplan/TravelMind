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
