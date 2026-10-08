# RAG Demo — Local RAG Pipeline with LangChain + LangGraph

A fully local Retrieval Augmented Generation (RAG) system built with
LangChain and LangGraph. No cloud API keys required — runs entirely
on your machine using Ollama.

## What it does

Takes a JSON knowledge base, embeds it into a vector store, and answers
questions grounded in that data. Refuses to answer questions outside
the knowledge base instead of hallucinating.

## Architecture

```
JSON data → Document Loader → Text Splitter → nomic-embed-text → FAISS
                                                                    ↓
User query → Embed query → FAISS similarity search → Grade chunks
                                                          ↓
                                              Relevant? → Generate → Answer
                                              Not relevant? → Retry → Max retries → "I don't know"
```

## Stack

| Component    | Tool                  | Why                                  |
| ------------ | --------------------- | ------------------------------------ |
| LLM          | phi3:mini via Ollama  | Local, no API key, good reasoning    |
| Embeddings   | nomic-embed-text      | Purpose-built for retrieval tasks    |
| Vector store | FAISS                 | In-memory, saved to disk, zero setup |
| Framework    | LangChain + LangGraph | Industry standard for LLM pipelines  |

## Key design decisions

**LLM-based relevance grader** — Uses an LLM to judge if retrieved chunks are relevant to the question, not just keyword overlap. Handles synonyms and paraphrases better than simple keyword matching.

**Query rewriting and LLM-based grader** — Rewriting failed queries and using an LLM to grade chunk relevance significantly improves retrieval accuracy, reducing false positives and ensuring only relevant context reaches the generator.

**HyDE-lite** — Each document includes hypothetical questions it answers, embedded alongside the content. Fixes query-document vocabulary mismatch.

**LangGraph state machine** — The pipeline is wrapped in a state machine with three nodes: retriever, grader, and generator. Irrelevant chunks trigger a retry loop (max 2 retries) before falling back to "I don't know."

**MMR retrieval** — Uses Maximal Marginal Relevance to prevent topic clustering in retrieved chunks.

**Structured logging** — All major steps (retrieval, grading, generation, rewriting) are logged for debugging and observability.

## Project structure

```
rag-demo/
├── data/sample.json              # knowledge base
├── config.py                     # centralized config
├── ingest.py                     # load → chunk → embed → store
├── query.py                      # simple linear RAG pipeline
├── rag.py                        # full pipeline with interactive mode
├── rag_graph.py                  # LangGraph version with grader + retry loop
├── evaluate.py                   # test suite for RAG accuracy
├── .env.example                  # example environment variables
├── .gitignore                    # ignored files (e.g., .env)
└── requirements.txt
```

## Setup

```bash
# Install Ollama and pull models
ollama pull phi3:mini
ollama pull nomic-embed-text

# (Optional) Create a .env file for model configuration
cp .env.example .env
# Edit .env to set your model paths and max retries

# Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Build vectorstore
python3 ingest.py

# Run simple pipeline
python3 rag.py

# Run LangGraph pipeline (with grader + retry)
python3 rag_graph.py

# Evaluate the pipeline
python3 evaluate.py
```

## What I learned building this

- Retrieval failures cause 90% of bad RAG answers — not the LLM
- FAISS scores are L2 distance (lower = better match)
- Chunk size directly affects embedding quality
- MMR prevents similar chunks from crowding out relevant ones
- LangGraph state machines let you add decision logic between pipeline steps
- Structured logging is essential for debugging complex pipelines
