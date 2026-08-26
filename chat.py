"""
chat.py — Ask questions about your own documents.

    python chat.py

Type a question, get an answer grounded in your docs/ folder, with
sources listed underneath. Type 'exit' or Ctrl+C to quit.
"""

import sys
import requests

import config
from retriever import load_index, retrieve


SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "context provided below, which comes from the user's own documents. "
    "If the answer isn't in the context, say you don't know rather than "
    "guessing. Keep answers concise and cite which source file you used "
    "when relevant."
)


def build_prompt(question, chunks):
    context_blocks = []
    for c in chunks:
        context_blocks.append(f"[Source: {c['source']}]\n{c['text']}")
    context = "\n\n---\n\n".join(context_blocks)

    return (
        f"Context from the user's documents:\n\n{context}\n\n"
        f"---\n\nQuestion: {question}"
    )


def generate_answer(question, chunks):
    headers = {
        "Authorization": f"Bearer {config.NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.CHAT_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, chunks)},
        ],
        "temperature": 0.3,
        "max_tokens": 800,
        "stream": True,
    }

    resp = requests.post(config.CHAT_URL, headers=headers, json=payload, stream=True, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Chat request failed ({resp.status_code}): {resp.text}")

    full_answer = ""
    for line in resp.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if not line.startswith("data: "):
            continue
        data = line[len("data: "):]
        if data.strip() == "[DONE]":
            break
        try:
            import json
            delta = json.loads(data)["choices"][0]["delta"].get("content", "")
        except (KeyError, IndexError, ValueError):
            continue
        print(delta, end="", flush=True)
        full_answer += delta
    print()
    return full_answer


def main():
    if not config.NVIDIA_API_KEY:
        sys.exit(
            "ERROR: NVIDIA_API_KEY is not set.\n"
            "Copy .env.example to .env and paste in your key from build.nvidia.com."
        )

    print("Loading index...")
    embeddings, metadata = load_index()
    print(f"Loaded {len(metadata)} chunks. Ask away (type 'exit' to quit).\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Bye!")
            break

        chunks = retrieve(question, embeddings, metadata)
        if not chunks:
            print("Assistant: I couldn't find anything relevant in your documents.\n")
            continue

        print("Assistant: ", end="", flush=True)
        generate_answer(question, chunks)

        sources = sorted(set(c["source"] for c in chunks))
        print(f"(Sources: {', '.join(sources)})\n")


if __name__ == "__main__":
    main()
