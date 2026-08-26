export interface ChunkMetadata {
  id: string;
  source: string;
  chunkIndex: number;
  text: string;
}

export interface ChunkWithEmbedding extends ChunkMetadata {
  embedding: number[];
}

export interface RerankedChunk {
  chunk: ChunkMetadata;
  score: number;
}

const CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions";
const EMBED_URL = "https://integrate.api.nvidia.com/v1/embeddings";
const RERANK_URL = "https://integrate.api.nvidia.com/v1/ranking";

const EMBED_MODEL = "nvidia/nv-embedqa-e5-v5";
const RERANK_MODEL = "nvidia/llama-nemotron-rerank-1b-v2";
const CHAT_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5";

const CHUNK_SIZE = 800;
const CHUNK_OVERLAP = 150;

export function getApiKey(): string {
  const key = process.env.NVIDIA_API_KEY || "";
  if (!key) {
    throw new Error("NVIDIA_API_KEY environment variable is not set.");
  }
  return key;
}

/**
 * Splits text into character-level overlapping chunks.
 */
export function chunkText(text: string, sourceName: string): ChunkMetadata[] {
  const chunks: ChunkMetadata[] = [];
  let start = 0;
  const n = text.length;
  let index = 0;

  while (start < n) {
    const end = Math.min(start + CHUNK_SIZE, n);
    const chunkStr = text.substring(start, end).trim();
    if (chunkStr) {
      chunks.push({
        id: `${sourceName}-${index}`,
        source: sourceName,
        chunkIndex: index,
        text: chunkStr,
      });
      index++;
    }
    if (end === n) break;
    start = end - CHUNK_OVERLAP;
  }
  return chunks;
}

/**
 * Call NVIDIA Embedding API for a batch of texts.
 * input_type: 'passage' (for document chunks) or 'query' (for search questions)
 */
export async function embedTexts(texts: string[], inputType: 'passage' | 'query'): Promise<number[][]> {
  const apiKey = getApiKey();
  const response = await fetch(EMBED_URL, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: EMBED_MODEL,
      input: texts,
      input_type: inputType,
      encoding_format: "float",
      truncate: "END",
    }),
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`Embedding request failed (${response.status}): ${errText}`);
  }

  const data = await response.json();
  return data.data.map((item: { embedding: number[] }) => item.embedding);
}

/**
 * Cosine similarity between two float vectors.
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

/**
 * Call NVIDIA Rerank API to re-score candidate chunks against the user question.
 */
export async function rerankChunks(query: string, candidates: ChunkMetadata[]): Promise<RerankedChunk[]> {
  if (candidates.length === 0) return [];
  const apiKey = getApiKey();

  const passages = candidates.map(c => ({ text: c.text }));

  const response = await fetch(RERANK_URL, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: RERANK_MODEL,
      query: { text: query },
      passages: passages,
      truncate: "END",
    }),
  });

  if (!response.ok) {
    const errText = await response.text();
    console.warn(`Reranking API warning (${response.status}): ${errText}. Falling back to cosine similarity order.`);
    return candidates.slice(0, 5).map(chunk => ({ chunk, score: 0.5 }));
  }

  const data = await response.json();
  const rankings: { index: number; logit?: number; score?: number }[] = data.rankings || data.data || [];

  const results: RerankedChunk[] = rankings.map(r => ({
    chunk: candidates[r.index],
    score: r.score ?? r.logit ?? 0,
  }));

  // Sort descending by score
  results.sort((a, b) => b.score - a.score);
  return results.slice(0, 5);
}

/**
 * Send the question + top reranked chunks to Nemotron Super 49B to generate the grounded answer.
 */
export async function generateAnswer(query: string, topChunks: RerankedChunk[]): Promise<{ answer: string; sources: string[] }> {
  const apiKey = getApiKey();

  const contextStr = topChunks
    .map((item, i) => `[Document ${i + 1}: ${item.chunk.source}]\n${item.chunk.text}`)
    .join("\n\n---\n\n");

  const systemPrompt = `You are a helpful assistant that answers user questions based strictly on the provided document excerpts.
Always be direct, concise, and truthful. If the documents do not contain enough information to answer, explicitly state that.
Cite your sources by mentioning the document filename(s) when answering.`;

  const userPrompt = `Context Documents:
${contextStr}

User Question: ${query}

Provide a comprehensive answer based ONLY on the context above:`;

  const response = await fetch(CHAT_URL, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: CHAT_MODEL,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
      temperature: 0.2,
      top_p: 0.7,
      max_tokens: 1024,
    }),
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`Generation request failed (${response.status}): ${errText}`);
  }

  const data = await response.json();
  const answer = data.choices[0]?.message?.content || "No response received.";
  const sources = Array.from(new Set(topChunks.map(c => c.chunk.source)));

  return { answer, sources };
}
