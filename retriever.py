"""
retriever.py — Loads the local index and answers "what's relevant?"

Two-stage retrieval:
  1. Embedding similarity (fast, approximate) narrows thousands of chunks
     down to a shortlist (TOP_K_CANDIDATES).
  2. Reranking (slower, precise) re-scores that shortlist and keeps the
     best few (TOP_K_FINAL) to actually show the LLM.
"""

import os
import sys
import json
import requests
import numpy as np

import config


def load_index():
    if not (os.path.exists(config.EMBEDDINGS_FILE) and os.path.exists(config.METADATA_FILE)):
        sys.exit(
            "No index found. Run 'python ingest.py' first to build one from your docs/ folder."
        )
    embeddings = np.load(config.EMBEDDINGS_FILE)
    with open(config.METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return embeddings, metadata


def embed_query(question):
    headers = {
        "Authorization": f"Bearer {config.NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.EMBED_MODEL,
        "input": [question],
        "input_type": "query",   # asymmetric embedding: query side
        "encoding_format": "float",
        "truncate": "END",
    }
    resp = requests.post(config.EMBED_URL, headers=headers, json=payload, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Embedding request failed ({resp.status_code}): {resp.text}")
    return np.array(resp.json()["data"][0]["embedding"], dtype=np.float32)


def cosine_similarity(query_vec, doc_matrix):
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    doc_norms = doc_matrix / (np.linalg.norm(doc_matrix, axis=1, keepdims=True) + 1e-8)
    return doc_norms @ query_norm


def rerank(question, candidates):
    """candidates: list of metadata dicts (each has a 'text' field).
    Returns the same dicts, re-ordered best-first, trimmed to TOP_K_FINAL.
    """
    headers = {
        "Authorization": f"Bearer {config.NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.RERANK_MODEL,
        "query": {"text": question},
        "passages": [{"text": c["text"]} for c in candidates],
        "truncate": "END",
    }
    resp = requests.post(config.RERANK_URL, headers=headers, json=payload, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Rerank request failed ({resp.status_code}): {resp.text}")

    rankings = resp.json()["rankings"]  # [{"index": N, "logit": F}, ...]
    rankings.sort(key=lambda r: r["logit"], reverse=True)
    ordered = [candidates[r["index"]] for r in rankings]
    return ordered[: config.TOP_K_FINAL]


def retrieve(question, embeddings, metadata):
    """Full pipeline: embed -> similarity shortlist -> rerank -> top chunks."""
    query_vec = embed_query(question)
    scores = cosine_similarity(query_vec, embeddings)

    top_k = min(config.TOP_K_CANDIDATES, len(metadata))
    top_indices = np.argsort(-scores)[:top_k]
    candidates = [metadata[i] for i in top_indices]

    if len(candidates) <= config.TOP_K_FINAL:
        return candidates  # too few candidates to bother reranking

    return rerank(question, candidates)
