# Nemotron Local RAG — ask questions about your own notes and PDFs

A small, fully local command-line app that lets you drop `.txt`, `.md`,
or `.pdf` files into a folder and then ask questions about them in
plain English. It uses three NVIDIA-hosted models (free tier) to do
the actual work — nothing runs on your GPU, and there's no vector
database to install.

```
docs/ (your files)  →  ingest.py  →  local index (data/)
                                          │
question ──────────────────────► chat.py │  → retrieves relevant chunks
                                          │  → reranks them
                                          └─► asks Nemotron to answer
```

---

## 1. Which Nemotron models this uses (and why three, not one)

RAG ("Retrieval-Augmented Generation") has three distinct jobs, and NVIDIA
has a separate model tuned for each one. Using the right specialist for
each job gives much better results than using one big model for
everything.

| Job | Model used in this project | What it does |
|---|---|---|
| **1. Embedding** | `nvidia/llama-3.2-nemoretriever-300m-embed-v2` | Converts each chunk of your documents (and later, your questions) into a vector of numbers, so "similar meaning" text ends up close together mathematically. This is what makes search possible. |
| **2. Reranking** | `nvidia/llama-nemotron-rerank-1b-v2` | Takes the ~20 chunks that looked similar by embedding and re-scores them more carefully against your exact question, keeping only the best ~5. Embedding search is fast but fuzzy; reranking is slower but much more accurate. |
| **3. Generation** | `nvidia/llama-3.3-nemotron-super-49b-v1.5` | The actual chat model. It reads your question plus the top chunks retrieval found, and writes an answer grounded in that context. |

**You only need to create ONE API key** — it's the same `nvapi-...` key
for every model in NVIDIA's catalog. You do need to visit each model's
page on build.nvidia.com at least once (see step 2) so your account has
access, but the key itself doesn't change per model.

You can swap the generation model in `config.py` if you want:
- `nvidia/nvidia-nemotron-nano-9b-v2` — smaller/faster/cheaper, good for quick tests
- `nvidia/llama-3.3-nemotron-super-49b-v1.5` — the default, stronger reasoning
- `nvidia/nemotron-3-ultra-550b-a55b` (if available on your account) — highest quality, slower

---

## 2. Get your free NVIDIA API key

1. Go to **https://build.nvidia.com** and sign in (or create an account —
   just an email, no credit card needed).
2. Search for each of these three models and open their page:
   - `llama-3.2-nemoretriever-300m-embed-v2`
   - `llama-nemotron-rerank-1b-v2`
   - `llama-3.3-nemotron-super-49b-v1.5`
3. On any model page, click **"Get API Key" → "Generate Key"**.
   The key starts with `nvapi-` and works across all models in the
   catalog — you don't need a separate key per model.
4. Copy the key somewhere safe. (Free-tier keys are rate-limited —
   roughly tens of requests per minute per model — and meant for
   prototyping, not production traffic.)

---

## 3. Setup

### Requirements
- Python 3.9+
- The packages in `requirements.txt`

### Install

```bash
cd nemotron-rag
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Add your API key

```bash
cp .env.example .env
```

Then open `.env` and paste your key:

```
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

## 4. Add your documents

Put your files directly inside the `docs/` folder. Supported types:

- `.txt` — plain text
- `.md` — markdown notes
- `.pdf` — text-based PDFs (scanned/image-only PDFs won't extract text;
  you'd need OCR first, which this project doesn't include)

```
nemotron-rag/
└── docs/
    ├── meeting_notes.md
    ├── research_paper.pdf
    └── project_ideas.txt
```

---

## 5. Build the index

Run this once after adding or changing files in `docs/`:

```bash
python ingest.py
```

What happens:
1. Reads every supported file in `docs/`
2. Splits each one into ~800-character overlapping chunks (so context
   isn't cut off mid-thought)
3. Sends chunks to the embedding model in small batches
4. Saves the resulting vectors (`data/embeddings.npy`) and the chunk
   text + source filenames (`data/metadata.json`) to disk

You'll see progress printed as it goes. Re-run this command any time
you add, remove, or edit files in `docs/` — it fully rebuilds the index.

---

## 6. Ask questions

```bash
python chat.py
```

Example session:

```
Loading index...
Loaded 47 chunks. Ask away (type 'exit' to quit).

You: What did we decide about the Q3 budget?
Assistant: According to meeting_notes.md, the team agreed to cap Q3
marketing spend at $40k and reallocate the remainder to engineering
hires...
(Sources: meeting_notes.md)

You: exit
Bye!
```

Each answer lists which of your source files it drew from, so you can
go verify anything that matters.

---

## 7. How it works under the hood (retriever.py)

1. **Embed the question** using the same embedding model, but with
   `input_type: "query"` instead of `"passage"` — these NeMo Retriever
   models are *asymmetric*, meaning questions and documents are encoded
   slightly differently for better matching.
2. **Cosine similarity** against every stored chunk vector (pure numpy,
   no external vector database needed at this scale) narrows things
   down to the top 20 candidates (`TOP_K_CANDIDATES` in `config.py`).
3. **Rerank** those 20 candidates by sending them plus the question to
   the reranking model. It returns a relevance score per chunk; we keep
   the top 5 (`TOP_K_FINAL`).
4. **Generate** the answer by sending the question and those 5 chunks
   to the chat model, with a system prompt instructing it to only
   answer from the given context.

---

## 8. Tuning knobs (all in `config.py`)

| Setting | Default | Effect of raising it |
|---|---|---|
| `CHUNK_SIZE` | 800 chars | Bigger chunks = more context per chunk, but less precise retrieval |
| `CHUNK_OVERLAP` | 150 chars | More overlap = less chance of splitting an idea across chunks, but more redundant storage |
| `TOP_K_CANDIDATES` | 20 | More candidates sent to reranking = slightly better recall, slower |
| `TOP_K_FINAL` | 5 | More chunks sent to the LLM = more context, but higher cost/latency and risk of noise |
| `CHAT_MODEL` | Nemotron Super 49B | Swap for Nano for speed, or a larger model for quality |

---

## 9. Troubleshooting

- **"NVIDIA_API_KEY is not set"** — you haven't created `.env`, or forgot
  to paste the key in. Check `cat .env` shows your real key.
- **401/403 errors** — your key may not have access to one of the three
  models yet. Visit that model's page on build.nvidia.com once while
  signed in to enable it for your account.
- **429 errors ("rate limited")** — the free tier caps requests per
  minute per model. Wait a few seconds and retry, or reduce
  `EMBED_BATCH_SIZE` in `config.py` if it happens during `ingest.py`.
  This is expected — the free tier is a prototyping tier, not an SLA.
- **PDF produces no text** — it's likely a scanned/image PDF. This
  project doesn't include OCR; you'd need to run it through an OCR
  tool first and save the output as `.txt`.
- **Answers ignore my documents** — rerun `python ingest.py` after
  adding files; `chat.py` only reads whatever is already indexed in
  `data/`.

---

## 10. Extending this project

Some natural next steps if you want to keep going:
- Swap the plain-text chunker for a smarter one that splits on
  paragraphs/headings instead of raw character counts.
- Add a simple web UI (Flask/FastAPI + a single HTML page) instead of
  the CLI.
- Add multimodal support using `nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1`
  and `nvidia/llama-nemotron-rerank-vl-1b-v2` to search over scanned
  pages, slides, and images directly instead of just extracted text.
- Persist conversation history so follow-up questions have context.
