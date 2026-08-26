import { NextRequest, NextResponse } from "next/server";
import { chunkText, embedTexts, ChunkWithEmbedding } from "@/lib/nemotron";

export async function POST(req: NextRequest) {
  try {
    const { documents } = await req.json();

    if (!documents || !Array.isArray(documents) || documents.length === 0) {
      return NextResponse.json(
        { error: "No documents provided. Expected array of { filename, text }." },
        { status: 400 }
      );
    }

    const allChunksWithEmbeddings: ChunkWithEmbedding[] = [];

    for (const doc of documents) {
      if (!doc.text || !doc.text.trim()) continue;
      const filename = doc.filename || "untitled.txt";
      const chunks = chunkText(doc.text, filename);

      // Embed chunks in batches of 16
      const BATCH_SIZE = 16;
      for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
        const batch = chunks.slice(i, i + BATCH_SIZE);
        const texts = batch.map(c => c.text);
        const embeddings = await embedTexts(texts, "passage");

        batch.forEach((chunk, idx) => {
          allChunksWithEmbeddings.push({
            ...chunk,
            embedding: embeddings[idx],
          });
        });
      }
    }

    return NextResponse.json({
      success: true,
      count: allChunksWithEmbeddings.length,
      chunks: allChunksWithEmbeddings,
    });
  } catch (err: any) {
    console.error("Ingestion error:", err);
    return NextResponse.json(
      { error: err.message || "Failed to process and embed documents." },
      { status: 500 }
    );
  }
}
