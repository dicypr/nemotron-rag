"use client";

import { useState, useRef, useEffect } from "react";
import {
  FileText,
  Upload,
  Send,
  Sparkles,
  Database,
  Cpu,
  Layers,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Trash2,
  BookOpen,
  ChevronDown,
  ChevronUp,
  Plus,
  RefreshCw,
} from "lucide-react";

interface DocumentItem {
  id: string;
  filename: string;
  text: string;
  charCount: number;
}

interface ChunkItem {
  id: string;
  source: string;
  chunkIndex: number;
  text: string;
  embedding: number[];
}

interface Message {
  id: string;
  sender: "user" | "assistant";
  text: string;
  sources?: string[];
  retrievedChunks?: { source: string; text: string; score: number }[];
  timestamp: string;
}

export default function Home() {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [chunks, setChunks] = useState<ChunkItem[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputQuery, setInputQuery] = useState("");
  
  // Status states
  const [isIngesting, setIsIngesting] = useState(false);
  const [isQuerying, setIsQuerying] = useState(false);
  const [statusStep, setStatusStep] = useState<string>("");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  
  // Text paste modal
  const [showPasteModal, setShowPasteModal] = useState(false);
  const [pasteTitle, setPasteTitle] = useState("");
  const [pasteContent, setPasteContent] = useState("");
  
  // Expanded retrieved sources state per message
  const [expandedSources, setExpandedSources] = useState<Record<string, boolean>>({});

  const chatEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isQuerying]);

  // Handle uploading files (.txt, .md, .pdf)
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const newDocs: DocumentItem[] = [];
    setErrorMessage(null);

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const filename = file.name;
      const lower = filename.toLowerCase();

      try {
        if (lower.endsWith(".pdf")) {
          const text = await file.text();
          newDocs.push({
            id: Math.random().toString(36).substr(2, 9),
            filename,
            text,
            charCount: text.length,
          });
        } else {
          const text = await file.text();
          newDocs.push({
            id: Math.random().toString(36).substr(2, 9),
            filename,
            text,
            charCount: text.length,
          });
        }
      } catch (err: any) {
        console.error(`Error reading ${filename}:`, err);
        setErrorMessage(`Could not read file ${filename}`);
      }
    }

    setDocuments((prev) => [...prev, ...newDocs]);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  // Add pasted text document
  const handleAddPastedDocument = () => {
    if (!pasteContent.trim()) return;
    const filename = pasteTitle.trim() ? (pasteTitle.endsWith(".txt") || pasteTitle.endsWith(".md") ? pasteTitle : `${pasteTitle}.txt`) : `note_${documents.length + 1}.txt`;
    setDocuments((prev) => [
      ...prev,
      {
        id: Math.random().toString(36).substr(2, 9),
        filename,
        text: pasteContent,
        charCount: pasteContent.length,
      },
    ]);
    setPasteTitle("");
    setPasteContent("");
    setShowPasteModal(false);
  };

  // Delete a document from list
  const handleDeleteDocument = (id: string) => {
    setDocuments((prev) => prev.filter((d) => d.id !== id));
  };

  // Load sample documents to quick test
  const handleLoadSampleDocs = () => {
    const sample1 = {
      id: "sample-1",
      filename: "nemotron_architecture.md",
      text: `# NVIDIA Nemotron RAG System Overview\n\nThe Nemotron RAG system combines three specialized NVIDIA NeMo Retriever and LLM models under one free API key from build.nvidia.com.\n\n1. **Embedding**: nvidia/llama-3.2-nemoretriever-300m-embed-v2 converts notes and PDFs into 1024-dimensional dense vectors.\n2. **Reranking**: nvidia/llama-nemotron-rerank-1b-v2 filters top match candidate vectors down to the most strictly relevant chunks.\n3. **Generation**: nvidia/llama-3.3-nemotron-super-49b-v1.5 synthesizes the final grounded answer with full reasoning capability.`,
      charCount: 620,
    };
    const sample2 = {
      id: "sample-2",
      filename: "q3_budget_report.txt",
      text: `Q3 Financial & Strategy Summary:\n- Cap Q3 marketing expenditure at $40,000.\n- Reallocate $25,000 remaining surplus directly to engineering staff hires.\n- Target launch date for cloud deployment is September 15.`,
      charCount: 220,
    };
    setDocuments([sample1, sample2]);
  };

  // Ingest documents -> Call API to chunk & embed
  const handleIngest = async () => {
    if (documents.length === 0) {
      setErrorMessage("Please upload or add at least one document to ingest.");
      return;
    }

    setIsIngesting(true);
    setErrorMessage(null);
    setStatusStep("Sending documents to NVIDIA NeMo Embedding Model...");

    try {
      const res = await fetch("/api/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ documents }),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "Ingestion failed.");
      }

      setChunks(data.chunks);
      setStatusStep("");
    } catch (err: any) {
      console.error(err);
      setErrorMessage(err.message || "An error occurred during vector ingestion.");
    } finally {
      setIsIngesting(false);
    }
  };

  // Chat Query -> Submit question
  const handleSendMessage = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!inputQuery.trim() || isQuerying) return;

    if (chunks.length === 0) {
      setErrorMessage("Please click 'Build Vector Index' to index your documents before asking questions.");
      return;
    }

    const userMsg: Message = {
      id: Math.random().toString(36).substr(2, 9),
      sender: "user",
      text: inputQuery.trim(),
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMsg]);
    const currentQuery = inputQuery;
    setInputQuery("");
    setIsQuerying(true);
    setErrorMessage(null);

    setStatusStep("Step 1/3: Embedding query vector...");

    try {
      setTimeout(() => setStatusStep("Step 2/3: NeMo 1B Reranking top 20 candidate chunks..."), 1200);
      setTimeout(() => setStatusStep("Step 3/3: Synthesizing answer with Nemotron Super 49B..."), 2500);

      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: currentQuery, chunks }),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "Failed to generate answer.");
      }

      const assistantMsg: Message = {
        id: Math.random().toString(36).substr(2, 9),
        sender: "assistant",
        text: data.answer,
        sources: data.sources,
        retrievedChunks: data.retrievedChunks,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err: any) {
      console.error(err);
      setErrorMessage(err.message || "Failed to retrieve answer from Nemotron.");
    } finally {
      setIsQuerying(false);
      setStatusStep("");
    }
  };

  const toggleSourceExpand = (id: string) => {
    setExpandedSources((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  return (
    <div className="flex flex-col h-screen w-full overflow-hidden bg-[#090d16]">
      {/* Top Navbar */}
      <header className="h-16 border-b border-gray-800/80 glass-panel px-6 flex items-center justify-between shrink-0 z-20">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-[#76b900] to-emerald-600 flex items-center justify-center text-black font-bold shadow-lg shadow-[#76b900]/20">
            <Sparkles className="w-5 h-5 text-black" />
          </div>
          <div>
            <h1 className="font-bold text-lg text-white flex items-center gap-2">
              Nemotron RAG <span className="text-xs px-2 py-0.5 rounded-full bg-[#76b900]/20 text-[#76b900] border border-[#76b900]/30 font-medium">Vercel Ready</span>
            </h1>
            <p className="text-xs text-gray-400">Powered by NVIDIA NeMo Retriever & Nemotron 3.3 Super 49B</p>
          </div>
        </div>

        {/* Model badges */}
        <div className="hidden md:flex items-center gap-3">
          <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-gray-900/70 border border-gray-800 text-xs text-gray-300">
            <Cpu className="w-3.5 h-3.5 text-[#76b900]" />
            <span>Embed: <strong className="text-white">300m-embed-v2</strong></span>
          </div>
          <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-gray-900/70 border border-gray-800 text-xs text-gray-300">
            <Layers className="w-3.5 h-3.5 text-indigo-400" />
            <span>Rerank: <strong className="text-white">rerank-1b-v2</strong></span>
          </div>
          <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-gray-900/70 border border-gray-800 text-xs text-gray-300">
            <Sparkles className="w-3.5 h-3.5 text-sky-400" />
            <span>Chat: <strong className="text-white">super-49b-v1.5</strong></span>
          </div>
        </div>
      </header>

      {/* Main Grid Body */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Sidebar - Document & Knowledge Base Management */}
        <aside className="w-80 md:w-96 border-r border-gray-800/80 bg-[#0d1322] flex flex-col shrink-0">
          <div className="p-4 border-b border-gray-800/80 flex items-center justify-between">
            <div className="flex items-center gap-2 text-gray-200 font-semibold text-sm">
              <Database className="w-4 h-4 text-[#76b900]" />
              <span>Document Store ({documents.length})</span>
            </div>
            <button
              onClick={handleLoadSampleDocs}
              className="text-xs text-gray-400 hover:text-[#76b900] transition flex items-center gap-1 bg-gray-900/60 px-2.5 py-1 rounded-md border border-gray-800"
              title="Load pre-made sample notes to quickly test search"
            >
              <RefreshCw className="w-3 h-3" />
              <span>Load Samples</span>
            </button>
          </div>

          {/* Action buttons */}
          <div className="p-4 flex flex-col gap-2.5 border-b border-gray-800/60 bg-gray-900/30">
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".txt,.md,.pdf"
              onChange={handleFileUpload}
              className="hidden"
            />
            
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={() => fileInputRef.current?.click()}
                className="flex items-center justify-center gap-1.5 text-xs font-medium bg-gray-800 hover:bg-gray-700 text-gray-200 py-2 px-3 rounded-lg border border-gray-700 transition"
              >
                <Upload className="w-3.5 h-3.5 text-[#76b900]" />
                <span>Upload Files</span>
              </button>
              
              <button
                onClick={() => setShowPasteModal(true)}
                className="flex items-center justify-center gap-1.5 text-xs font-medium bg-gray-800 hover:bg-gray-700 text-gray-200 py-2 px-3 rounded-lg border border-gray-700 transition"
              >
                <Plus className="w-3.5 h-3.5 text-sky-400" />
                <span>Paste Text</span>
              </button>
            </div>

            {/* Ingest button */}
            <button
              onClick={handleIngest}
              disabled={isIngesting || documents.length === 0}
              className="w-full mt-1 flex items-center justify-center gap-2 bg-gradient-to-r from-[#76b900] to-emerald-600 hover:from-emerald-500 hover:to-[#76b900] text-black font-semibold py-2.5 px-4 rounded-lg shadow-md transition disabled:opacity-50 disabled:cursor-not-allowed text-xs"
            >
              {isIngesting ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin text-black" />
                  <span>Embedding Vectors...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4 text-black" />
                  <span>Build Vector Index ({chunks.length} Chunks)</span>
                </>
              )}
            </button>
          </div>

          {/* Documents List */}
          <div className="flex-1 overflow-y-auto p-4 space-y-2">
            {documents.length === 0 ? (
              <div className="text-center py-10 px-4 text-gray-500 border border-dashed border-gray-800 rounded-xl bg-gray-900/20">
                <FileText className="w-8 h-8 mx-auto mb-2 text-gray-600 stroke-[1.5]" />
                <p className="text-xs font-medium text-gray-400">No documents added yet</p>
                <p className="text-[11px] text-gray-600 mt-1">Upload `.txt`, `.md`, or `.pdf` files, or click "Load Samples" above.</p>
              </div>
            ) : (
              documents.map((doc) => (
                <div
                  key={doc.id}
                  className="p-3 rounded-xl bg-gray-900/70 border border-gray-800 hover:border-gray-700 flex items-center justify-between group transition"
                >
                  <div className="flex items-center gap-2.5 min-w-0">
                    <FileText className="w-4 h-4 text-[#76b900] shrink-0" />
                    <div className="min-w-0">
                      <p className="text-xs font-medium text-gray-200 truncate">{doc.filename}</p>
                      <p className="text-[10px] text-gray-500">{doc.charCount.toLocaleString()} chars</p>
                    </div>
                  </div>
                  <button
                    onClick={() => handleDeleteDocument(doc.id)}
                    className="opacity-0 group-hover:opacity-100 p-1 text-gray-500 hover:text-red-400 transition"
                    title="Remove document"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))
            )}
          </div>

          {/* Index Status Footer */}
          <div className="p-4 border-t border-gray-800/80 bg-gray-950/60 text-xs">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-gray-400">Index Status:</span>
              {chunks.length > 0 ? (
                <span className="flex items-center gap-1 text-[#76b900] font-medium">
                  <CheckCircle2 className="w-3.5 h-3.5" /> Ready ({chunks.length} chunks)
                </span>
              ) : (
                <span className="text-amber-400 font-medium">Not Indexed</span>
              )}
            </div>
            <p className="text-[11px] text-gray-500">
              {chunks.length > 0
                ? "Vector embeddings stored in session memory ready for RAG query."
                : "Add documents and click 'Build Vector Index' to begin querying."}
            </p>
          </div>
        </aside>

        {/* Right Main Chat Interface */}
        <main className="flex-1 flex flex-col overflow-hidden bg-[#090d16] relative">
          {/* Error Banner */}
          {errorMessage && (
            <div className="bg-red-950/80 border-b border-red-800/80 px-6 py-3 text-red-200 text-xs flex items-center justify-between shrink-0">
              <div className="flex items-center gap-2">
                <AlertCircle className="w-4 h-4 text-red-400 shrink-0" />
                <span>{errorMessage}</span>
              </div>
              <button
                onClick={() => setErrorMessage(null)}
                className="text-red-400 hover:text-red-200 font-bold ml-4"
              >
                ✕
              </button>
            </div>
          )}

          {/* Messages Area */}
          <div className="flex-1 overflow-y-auto p-6 space-y-6">
            {messages.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-center max-w-xl mx-auto px-4 py-12">
                <div className="w-14 h-14 rounded-2xl bg-[#76b900]/10 border border-[#76b900]/30 flex items-center justify-center text-[#76b900] mb-4 nvidia-glow">
                  <Sparkles className="w-7 h-7" />
                </div>
                <h2 className="text-2xl font-bold text-white mb-2">
                  Ask Nemotron About Your Documents
                </h2>
                <p className="text-sm text-gray-400 mb-8 leading-relaxed">
                  Drop your notes or PDFs into the sidebar, click <strong className="text-white">Build Vector Index</strong>, then ask questions here. NVIDIA NeMo will retrieve, rerank, and synthesize your answers.
                </p>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-3 w-full text-left">
                  <div className="p-3.5 rounded-xl glass-card text-xs">
                    <p className="font-semibold text-white mb-1 flex items-center gap-1.5">
                      <Cpu className="w-3.5 h-3.5 text-[#76b900]" /> 1. Vector Search
                    </p>
                    <p className="text-gray-400">Embeds question to pull top 20 relevant passage candidates.</p>
                  </div>
                  <div className="p-3.5 rounded-xl glass-card text-xs">
                    <p className="font-semibold text-white mb-1 flex items-center gap-1.5">
                      <Layers className="w-3.5 h-3.5 text-indigo-400" /> 2. NeMo Reranking
                    </p>
                    <p className="text-gray-400">Filters top 20 matches down to the 5 most accurate excerpts.</p>
                  </div>
                  <div className="p-3.5 rounded-xl glass-card text-xs">
                    <p className="font-semibold text-white mb-1 flex items-center gap-1.5">
                      <Sparkles className="w-3.5 h-3.5 text-sky-400" /> 3. 49B Answer
                    </p>
                    <p className="text-gray-400">Generates precise answer grounded strictly in retrieved context.</p>
                  </div>
                </div>
              </div>
            ) : (
              messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex flex-col ${msg.sender === "user" ? "items-end" : "items-start"}`}
                >
                  <div
                    className={`max-w-3xl rounded-2xl p-4 text-sm leading-relaxed ${
                      msg.sender === "user"
                        ? "bg-[#76b900]/15 text-white border border-[#76b900]/30 rounded-tr-none"
                        : "bg-gray-900/90 text-gray-100 border border-gray-800 rounded-tl-none shadow-xl"
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2 pb-1 border-b border-white/5 text-[11px] text-gray-400">
                      <span className="font-semibold flex items-center gap-1.5">
                        {msg.sender === "user" ? (
                          "You"
                        ) : (
                          <>
                            <Sparkles className="w-3 h-3 text-[#76b900]" /> Nemotron 3.3 Super 49B
                          </>
                        )}
                      </span>
                      <span>{msg.timestamp}</span>
                    </div>

                    <div className="whitespace-pre-wrap">{msg.text}</div>

                    {/* Sources & Reranker Breakdown */}
                    {msg.sender === "assistant" && msg.retrievedChunks && msg.retrievedChunks.length > 0 && (
                      <div className="mt-4 pt-3 border-t border-gray-800">
                        <div className="flex items-center justify-between text-xs text-gray-400">
                          <span className="flex items-center gap-1.5 font-medium text-gray-300">
                            <BookOpen className="w-3.5 h-3.5 text-[#76b900]" />
                            Sources: {msg.sources?.join(", ")}
                          </span>
                          <button
                            onClick={() => toggleSourceExpand(msg.id)}
                            className="flex items-center gap-1 text-[#76b900] hover:underline text-[11px]"
                          >
                            <span>{expandedSources[msg.id] ? "Hide Reranked Context" : "View Reranked Context"}</span>
                            {expandedSources[msg.id] ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                          </button>
                        </div>

                        {/* Expanded chunks list */}
                        {expandedSources[msg.id] && (
                          <div className="mt-3 space-y-2">
                            {msg.retrievedChunks.map((chunk, idx) => (
                              <div
                                key={idx}
                                className="p-3 rounded-lg bg-gray-950/80 border border-gray-800/80 text-xs"
                              >
                                <div className="flex items-center justify-between text-[11px] text-gray-400 mb-1">
                                  <span className="font-semibold text-gray-300">[{chunk.source}]</span>
                                  <span className="px-2 py-0.5 rounded bg-indigo-950 text-indigo-300 font-mono text-[10px] border border-indigo-800/50">
                                    Score: {chunk.score.toFixed(4)}
                                  </span>
                                </div>
                                <p className="text-gray-300 italic text-[11px] leading-relaxed">"{chunk.text}"</p>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}

            {/* Querying Status Indicator */}
            {isQuerying && (
              <div className="flex items-center gap-3 text-xs text-[#76b900] bg-[#76b900]/10 border border-[#76b900]/20 px-4 py-3 rounded-xl max-w-md">
                <Loader2 className="w-4 h-4 animate-spin text-[#76b900]" />
                <span>{statusStep || "Processing query through NeMo RAG pipeline..."}</span>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          {/* Input Bar Footer */}
          <div className="p-4 border-t border-gray-800/80 glass-panel shrink-0">
            <form onSubmit={handleSendMessage} className="flex items-center gap-2 max-w-4xl mx-auto">
              <input
                type="text"
                value={inputQuery}
                onChange={(e) => setInputQuery(e.target.value)}
                placeholder={
                  chunks.length === 0
                    ? "Add & index documents in the sidebar first..."
                    : "Ask anything about your documents..."
                }
                disabled={isQuerying}
                className="flex-1 bg-gray-900/90 text-gray-100 placeholder-gray-500 text-sm px-4 py-3 rounded-xl border border-gray-800 focus:outline-none focus:border-[#76b900] focus:ring-1 focus:ring-[#76b900] transition disabled:opacity-50"
              />
              <button
                type="submit"
                disabled={isQuerying || !inputQuery.trim()}
                className="bg-gradient-to-r from-[#76b900] to-emerald-600 hover:from-emerald-500 hover:to-[#76b900] text-black p-3 rounded-xl transition shadow-lg disabled:opacity-40 disabled:cursor-not-allowed shrink-0"
              >
                <Send className="w-4 h-4" />
              </button>
            </form>
          </div>
        </main>
      </div>

      {/* Paste Modal */}
      {showPasteModal && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-[#0e1424] border border-gray-800 rounded-2xl w-full max-w-lg p-6 shadow-2xl space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                <Plus className="w-4 h-4 text-[#76b900]" /> Paste Text Note
              </h3>
              <button
                onClick={() => setShowPasteModal(false)}
                className="text-gray-400 hover:text-white text-sm font-bold"
              >
                ✕
              </button>
            </div>

            <div className="space-y-3">
              <div>
                <label className="text-xs text-gray-400 block mb-1">Document Title (e.g. meeting_notes.md)</label>
                <input
                  type="text"
                  value={pasteTitle}
                  onChange={(e) => setPasteTitle(e.target.value)}
                  placeholder="notes.txt"
                  className="w-full bg-gray-900 border border-gray-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-[#76b900]"
                />
              </div>

              <div>
                <label className="text-xs text-gray-400 block mb-1">Text Content</label>
                <textarea
                  rows={6}
                  value={pasteContent}
                  onChange={(e) => setPasteContent(e.target.value)}
                  placeholder="Paste your text content here..."
                  className="w-full bg-gray-900 border border-gray-800 rounded-lg p-3 text-xs text-white focus:outline-none focus:border-[#76b900]"
                />
              </div>
            </div>

            <div className="flex items-center justify-end gap-2 pt-2">
              <button
                onClick={() => setShowPasteModal(false)}
                className="px-4 py-2 rounded-lg bg-gray-800 text-xs text-gray-300 hover:bg-gray-700"
              >
                Cancel
              </button>
              <button
                onClick={handleAddPastedDocument}
                disabled={!pasteContent.trim()}
                className="px-4 py-2 rounded-lg bg-[#76b900] text-xs text-black font-semibold hover:bg-emerald-400 disabled:opacity-50"
              >
                Add Document
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
