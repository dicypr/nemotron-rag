import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Nemotron Local RAG — AI Powered Document Intelligence",
  description: "Retrieve, rerank, and synthesize answers from your documents powered by NVIDIA Nemotron NeMo models.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased min-h-screen bg-[#090d16] text-gray-100 flex flex-col">
        {children}
      </body>
    </html>
  );
}
