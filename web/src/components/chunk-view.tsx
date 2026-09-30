"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { api, ApiError, type Chunk } from "@/lib/api";
import { CodeBlock, Notice } from "./console";
import { Header } from "./header";
import { Card, Eyebrow } from "./ui";

/** One chunk in full: source, the ordered calls behind structural answers, and provenance. */
export function ChunkView({ chunkId, version }: { chunkId: string; version?: string }) {
  const [loaded, setLoaded] = useState<{ key: string; chunk: Chunk } | null>(null);
  const [error, setError] = useState<{ key: string; message: string } | null>(null);
  const key = `${chunkId}@${version ?? ""}`;

  useEffect(() => {
    let cancelled = false;
    api
      .chunk(chunkId, version)
      .then((chunk) => !cancelled && setLoaded({ key, chunk }))
      .catch(
        (caught) =>
          !cancelled &&
          setError({ key, message: caught instanceof ApiError ? caught.message : "Could not load this chunk." }),
      );
    return () => {
      cancelled = true;
    };
  }, [chunkId, version, key]);

  const chunk = loaded?.key === key ? loaded.chunk : null;
  const failure = error?.key === key ? error.message : null;

  return (
    <>
      <Header />
      <div className="bg-void text-bone min-h-[calc(100dvh-4rem)]">
        <div className="container-page py-10 lg:py-14">
          <Link href="/search" className="text-mono-s uppercase opacity-70 transition-opacity hover:opacity-100">
            ← Back to the console
          </Link>
          {failure ? (
            <div className="mt-8">
              <Notice title="Chunk unavailable">{failure}</Notice>
            </div>
          ) : !chunk ? (
            <div
              className="mt-8 h-96 animate-pulse rounded-sm border border-white/8 bg-white/[0.03]"
              aria-busy="true"
            />
          ) : (
            <ChunkDetail chunk={chunk} />
          )}
        </div>
      </div>
    </>
  );
}

function ChunkDetail({ chunk }: { chunk: Chunk }) {
  const { location, metadata } = chunk;
  const lines = location.end_line - location.start_line + 1;
  return (
    <div className="mt-8">
      <Eyebrow>
        {metadata.kind} · {metadata.version_id}
      </Eyebrow>
      <h1 className="text-heading-40 mt-4 break-words">{metadata.symbol ?? "Module-level block"}</h1>
      <p className="text-mono-m mt-3 break-all opacity-75">
        {location.file_path}:{location.start_line}-{location.end_line}
      </p>

      <div className="mt-10 grid gap-6 lg:grid-cols-[1fr_20rem]">
        <Card className="min-w-0 border-white/12 bg-[#120e0b] [--frame-dot:var(--color-stroke-3)]">
          <div className="flex items-center justify-between border-b border-white/8 px-5 py-3.5 md:px-6">
            <span className="text-mono-s uppercase opacity-70">Source</span>
            <span className="text-mono-s opacity-70">
              {lines} line{lines === 1 ? "" : "s"} · {metadata.language}
            </span>
          </div>
          <CodeBlock text={chunk.text} startLine={location.start_line} maxLines={400} />
        </Card>

        <aside className="space-y-4">
          <Section title="Calls, in source order">
            {metadata.calls.length ? (
              <ol className="space-y-1.5">
                {metadata.calls.map((call, index) => (
                  <li key={`${call}-${index}`} className="text-mono-m flex gap-3">
                    <span className="text-sun w-6 shrink-0 text-right">{index}</span>
                    <span className="text-desert break-all">{call}</span>
                  </li>
                ))}
              </ol>
            ) : (
              <p className="text-body-14 opacity-70">No calls.</p>
            )}
            <p className="text-body-14 mt-3 opacity-70">
              The ordinal is what answers “calls X before Y”: it is read from source order at index time.
            </p>
          </Section>
          <Section title="Imports">
            {metadata.imports.length ? (
              <ul className="space-y-1.5">
                {metadata.imports.map((item) => (
                  <li key={item} className="text-mono-m break-all opacity-85">
                    {item}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-body-14 opacity-70">None in this chunk.</p>
            )}
          </Section>
          {metadata.docstring && (
            <Section title="Docstring">
              <p className="text-body-14 whitespace-pre-line opacity-85">{metadata.docstring}</p>
            </Section>
          )}
          <Section title="Provenance">
            <dl className="text-mono-s space-y-2.5">
              <Row label="Exported" value={metadata.is_exported ? "yes" : "no"} />
              {metadata.parent_symbol && <Row label="Parent" value={metadata.parent_symbol} />}
              <Row label="Commit" value={metadata.commit_sha ? metadata.commit_sha.slice(0, 12) : "—"} />
              <Row label="Content hash" value={chunk.content_hash} />
              <Row label="Chunk id" value={chunk.chunk_id} />
            </dl>
          </Section>
        </aside>
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-sm border border-white/10 bg-white/[0.02] p-5">
      <h2 className="text-mono-s mb-3 uppercase opacity-70">{title}</h2>
      {children}
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="uppercase opacity-60">{label}</dt>
      <dd className="mt-0.5 break-all opacity-90">{value}</dd>
    </div>
  );
}
