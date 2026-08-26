"""
Central configuration for the Nemotron Local RAG mini-app.

All three NVIDIA-hosted models used by this project are declared here.
You only need ONE API key (from build.nvidia.com) — it works for all
models in NVIDIA's catalog, not just one.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")

# Base URLs (NVIDIA's OpenAI-compatible + native ranking endpoints)
CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
EMBED_URL = "https://integrate.api.nvidia.com/v1/embeddings"
RERANK_URL = "https://integrate.api.nvidia.com/v1/ranking"

# ---------------------------------------------------------------------------
# Models — get an API key for each on build.nvidia.com (same key works for all)
# ---------------------------------------------------------------------------
# 1. Embedding model — turns text chunks into vectors for similarity search.
EMBED_MODEL = "nvidia/llama-3.2-nemoretriever-300m-embed-v2"

# 2. Reranker model — re-scores the top candidates for real relevance.
RERANK_MODEL = "nvidia/llama-nemotron-rerank-1b-v2"

# 3. Generator model — writes the final answer grounded in retrieved chunks.
#    Swap for "nvidia/nvidia-nemotron-nano-9b-v2" if you want faster/cheaper
#    answers, or "nvidia/llama-3.3-nemotron-super-49b-v1.5" (default) for
#    stronger reasoning.
CHAT_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # characters shared between consecutive chunks

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K_CANDIDATES = 20  # how many chunks to pull via embedding similarity
TOP_K_FINAL = 5        # how many chunks survive reranking and go to the LLM

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DOCS_DIR = "docs"
INDEX_DIR = "data"
EMBEDDINGS_FILE = os.path.join(INDEX_DIR, "embeddings.npy")
METADATA_FILE = os.path.join(INDEX_DIR, "metadata.json")

# Embedding batch size per API call (keeps requests small and reliable)
EMBED_BATCH_SIZE = 16
