import { NextRequest, NextResponse } from "next/server";
import {
  embedTexts,
  cosineSimilarity,
  rerankChunks,
  generateAnswer,
  ChunkWithEmbedding,
} from "@/lib/nemotron";

export async function POST(req: NextRequest) {
  try {
    const { question, chunks } = await req.json();

    if (!question || typeof question !== "string" || !question.trim()) {
      return NextResponse.json({ error: "Question is required." }, { status: 400 });
    }

    if (!chunks || !Array.isArray(chunks) || chunks.length === 0) {
      return NextResponse.json(
        { error: "No indexed document chunks provided. Please ingest documents first." },
        { status: 400 }
      );
    }

    // Step 1: Embed the user's question with input_type = 'query'
    const [queryEmbedding] = await embedTexts([question], "query");

    // Step 2: Compute cosine similarity against all indexed chunk vectors
    const scoredCandidates = (chunks as ChunkWithEmbedding[]).map(chunk => ({
      chunk,
      similarity: cosineSimilarity(queryEmbedding, chunk.embedding),
    }));

    // Sort by cosine similarity descending & take top 20 candidates
    scoredCandidates.sort((a, b) => b.similarity - a.similarity);
    const top20Candidates = scoredCandidates.slice(0, 20).map(sc => sc.chunk);

    // Step 3: Rerank the top 20 candidates using Nemotron Rerank 1B
    const top5Reranked = await rerankChunks(question, top20Candidates);

    // Step 4: Generate the answer using Nemotron 3.3 Super 49B
    const { answer, sources } = await generateAnswer(question, top5Reranked);

    return NextResponse.json({
      success: true,
      answer,
      sources,
      retrievedChunks: top5Reranked.map(r => ({
        source: r.chunk.source,
        text: r.chunk.text,
        score: r.score,
      })),
    });
  } catch (err: any) {
    console.error("Chat API error:", err);
    return NextResponse.json(
      { error: err.message || "Failed to process chat query." },
      { status: 500 }
    );
  }
}
