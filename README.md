# TED Talk RAG Assistant

This project is an AI assistant designed to answer questions about TED talks using a Retrieval-Augmented Generation (RAG) system. It uses FastAPI for the backend, Pinecone as a vector database, and OpenAI models for embeddings and text generation.

# Project Specifications

1. Data Processing and Efficiency
The assistant is indexed with the full ted_talks_en.csv dataset, covering approximately 4,005 talks. A total of 14,676 vectors are stored in Pinecone. To stay within the 5 USD budget and ensure stability, the ingestion process used batching with a size of 100 vectors per batch. This optimized API costs and prevented rate-limit errors.

2. RAG Hyperparameters
The following parameters are reported via the /api/stats endpoint:
- Chunk Size: 1024 tokens (approximately 4000 characters). This size balances semantic richness with prompt efficiency.
- Overlap Ratio: 0.2 (20%). This ensures context continuity and prevents information loss at the boundaries of text splits.
- Top-k: 8. I chose a value of 8 to ensure high recall, especially for questions requiring multiple talk recommendations. This allows the model to identify at least three unique talk titles even when multiple chunks originate from the same talk.

3. System Guardrails
The system is strictly grounded in the provided TED context. If an answer cannot be determined from the retrieved data, the model is instructed to respond: "I don't know based on the provided TED data."