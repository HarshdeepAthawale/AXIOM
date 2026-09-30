import type { Metadata } from "next";

import { ChunkView } from "@/components/chunk-view";

export const metadata: Metadata = {
  title: "Chunk · Axiom",
};

export default async function ChunkPage({
  params,
  searchParams,
}: {
  params: Promise<{ id: string }>;
  searchParams: Promise<{ version?: string }>;
}) {
  const { id } = await params;
  const { version } = await searchParams;
  return <ChunkView chunkId={id} version={version} />;
}
