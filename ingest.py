"""
ingest.py — Build (or rebuild) the local search index.

Run this once whenever you add/change files in docs/:

    python ingest.py

It will:
  1. Read every .txt, .md, and .pdf file under docs/
  2. Split each file into overlapping chunks
  3. Send chunks to NVIDIA's Nemotron embedding model in small batches
  4. Save the resulting vectors + metadata to data/
"""

import os
import sys
import json
import requests
import numpy as np
from pypdf import PdfReader

import config


def read_text_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf_file(path):
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def load_documents(docs_dir):
    """Return a list of (filename, full_text) for every supported file."""
    documents = []
    if not os.path.isdir(docs_dir):
        print(f"No '{docs_dir}/' folder found. Create it and add your notes/PDFs.")
        return documents

    for name in sorted(os.listdir(docs_dir)):
        path = os.path.join(docs_dir, name)
        if not os.path.isfile(path):
            continue
        lower = name.lower()
        try:
            if lower.endswith((".txt", ".md")):
                text = read_text_file(path)
            elif lower.endswith(".pdf"):
                text = read_pdf_file(path)
            else:
                continue
        except Exception as e:
            print(f"  ! Skipping {name} (couldn't read it: {e})")
            continue

        if text.strip():
            documents.append((name, text))
            print(f"  + Loaded {name} ({len(text)} chars)")
        else:
            print(f"  ! {name} produced no extractable text — skipping")

    return documents


def chunk_text(text, chunk_size, overlap):
    """Simple sliding-window character chunker."""
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap  # step forward, keeping some overlap
    return chunks


def embed_batch(texts, input_type):
    """Call NVIDIA's embedding endpoint for a batch of texts.

    input_type must be 'passage' (for document chunks) or 'query'
    (for user questions) — the NeMo Retriever embedding models are
    asymmetric and use this to encode each side differently.
    """
    headers = {
        "Authorization": f"Bearer {config.NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.EMBED_MODEL,
        "input": texts,
        "input_type": input_type,
        "encoding_format": "float",
        "truncate": "END",
    }
    resp = requests.post(config.EMBED_URL, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Embedding request failed ({resp.status_code}): {resp.text}")

    data = resp.json()["data"]
    # Results are returned in the same order as the input list
    return [item["embedding"] for item in data]


def build_index():
    if not config.NVIDIA_API_KEY:
        sys.exit(
            "ERROR: NVIDIA_API_KEY is not set.\n"
            "Copy .env.example to .env and paste in your key from build.nvidia.com."
        )

    print(f"Scanning '{config.DOCS_DIR}/' for documents...")
    documents = load_documents(config.DOCS_DIR)
    if not documents:
        sys.exit("No documents found. Add .txt, .md, or .pdf files to docs/ and re-run.")

    # Build all chunks with metadata pointing back to their source file
    all_chunks = []       # list[str]
    all_metadata = []      # list[dict]
    for filename, text in documents:
        chunks = chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({"source": filename, "chunk_index": i, "text": chunk})

    print(f"\nCreated {len(all_chunks)} chunks from {len(documents)} document(s).")
    print(f"Embedding with {config.EMBED_MODEL} ...")

    all_embeddings = []
    batch_size = config.EMBED_BATCH_SIZE
    for start in range(0, len(all_chunks), batch_size):
        batch = all_chunks[start:start + batch_size]
        vectors = embed_batch(batch, input_type="passage")
        all_embeddings.extend(vectors)
        print(f"  embedded {min(start + batch_size, len(all_chunks))}/{len(all_chunks)}")

    embeddings_array = np.array(all_embeddings, dtype=np.float32)

    os.makedirs(config.INDEX_DIR, exist_ok=True)
    np.save(config.EMBEDDINGS_FILE, embeddings_array)
    with open(config.METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Index saved to '{config.INDEX_DIR}/'.")
    print(f"  - {config.EMBEDDINGS_FILE}  ({embeddings_array.shape})")
    print(f"  - {config.METADATA_FILE}")
    print("\nYou can now run: python chat.py")


if __name__ == "__main__":
    build_index()
