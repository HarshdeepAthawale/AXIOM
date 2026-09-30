"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useState, type FormEvent } from "react";

import {
  api,
  ApiError,
  SIGNAL_COLOR,
  SIGNAL_LABEL,
  type Family,
  type Health,
  type QueryResponse,
  type QueryType,
  type Result,
  type SignalName,
  type VersionsResponse,
} from "@/lib/api";
import { codeLines, diffLineKind, locationRef, SIGNALS, splitReason, weightShares } from "@/lib/format";
import { Header } from "./header";
import { Card, cx, Eyebrow } from "./ui";

const EXAMPLES: Array<{ label: string; query: string }> = [
  { label: "Structural", query: "Which files call parseIntent before resolveTool?" },
  { label: "Usage", query: "Where is the bluetooth-settings deeplink used?" },
  { label: "Semantic", query: "How is user input normalized before dispatch?" },
];

const QUERY_TYPES: Array<{ value: "" | QueryType; label: string }> = [
  { value: "", label: "Auto (classifier)" },
  { value: "semantic", label: "Semantic" },
  { value: "structural", label: "Structural" },
  { value: "usage", label: "Usage" },
  { value: "hybrid", label: "Hybrid" },
];

export function Console() {
  const params = useSearchParams();
  const router = useRouter();
  const tab = params.get("tab") === "families" ? "families" : "search";

  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [versions, setVersions] = useState<VersionsResponse | null>(null);
  const [version, setVersion] = useState<string>("");
  const [allVersions, setAllVersions] = useState(false);
  const [topK, setTopK] = useState(8);
  const [queryType, setQueryType] = useState<"" | QueryType>("");

  const refreshHealth = useCallback(async () => {
    try {
      setHealth(await api.health());
      setHealthError(null);
    } catch (error) {
      setHealth(null);
      setHealthError(error instanceof ApiError ? error.message : "The Axiom API is not reachable.");
    }
  }, []);

  useEffect(() => {
    const first = setTimeout(refreshHealth, 0);
    const timer = setInterval(refreshHealth, 15000);
    return () => {
      clearTimeout(first);
      clearInterval(timer);
    };
  }, [refreshHealth]);

  useEffect(() => {
    if (!health?.index_available) return;
    api
      .versions()
      .then(setVersions)
      .catch(() => setVersions(null));
  }, [health?.index_available]);

  const setTab = (next: "search" | "families") =>
    router.replace(next === "families" ? "/search?tab=families" : "/search", { scroll: false });

  return (
    <>
      <Header right={<StatusPill health={health} error={healthError} />} />
      <div className="bg-void min-h-[calc(100dvh-4rem)] text-bone">
        <div className="container-page grid gap-8 py-8 lg:grid-cols-[17rem_1fr] lg:gap-12 lg:py-12">
          <aside className="space-y-8 max-lg:order-2 lg:sticky lg:top-24 lg:self-start">
            <div>
              <Eyebrow>Console</Eyebrow>
              <h1 className="text-heading-32 mt-4">Ask the codebase</h1>
            </div>
            <div className="space-y-5 border-t border-white/10 pt-6">
              <Field label="Version">
                <select
                  value={version}
                  onChange={(event) => setVersion(event.target.value)}
                  disabled={allVersions}
                  className="text-body-14 w-full rounded-md border border-white/12 bg-[#15110e] px-3 py-2 disabled:opacity-55"
                >
                  <option value="">Active{versions?.active_version ? ` (${versions.active_version})` : ""}</option>
                  {versions?.versions.map((info) => (
                    <option key={info.version_id} value={info.version_id}>
                      {info.version_id} · {info.chunk_count} chunks
                    </option>
                  ))}
                </select>
              </Field>
              <label className="flex cursor-pointer items-start gap-3">
                <input
                  type="checkbox"
                  checked={allVersions}
                  onChange={(event) => setAllVersions(event.target.checked)}
                  className="accent-sun mt-1"
                />
                <span>
                  <span className="text-body-14 block">Search every version</span>
                  <span className="text-body-14 block opacity-60">Near-identical snippets collapse into families.</span>
                </span>
              </label>
              <ResultCount value={topK} onChange={setTopK} />
              <Field label="Query type">
                <select
                  value={queryType}
                  onChange={(event) => setQueryType(event.target.value as "" | QueryType)}
                  className="text-body-14 w-full rounded-md border border-white/12 bg-[#15110e] px-3 py-2"
                >
                  {QUERY_TYPES.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </Field>
            </div>
            {health && (
              <dl className="text-mono-s space-y-2 border-t border-white/10 pt-6 uppercase opacity-60">
                <div className="flex justify-between gap-4">
                  <dt>Profile</dt>
                  <dd>{health.profile}</dd>
                </div>
                <div className="flex justify-between gap-4">
                  <dt>Versions</dt>
                  <dd>{versions?.versions.length ?? "—"}</dd>
                </div>
                <div className="flex justify-between gap-4">
                  <dt>Axiom</dt>
                  <dd>{health.version}</dd>
                </div>
              </dl>
            )}
          </aside>

          <main className="min-w-0">
            <div className="mb-8 flex gap-6 border-b border-white/10" role="tablist">
              {(["search", "families"] as const).map((name) => (
                <button
                  key={name}
                  role="tab"
                  aria-selected={tab === name}
                  onClick={() => setTab(name)}
                  className={cx(
                    "text-body-14 -mb-px cursor-pointer border-b-2 pb-3 transition-colors",
                    tab === name ? "border-sun" : "border-transparent opacity-60 hover:opacity-100",
                  )}
                >
                  {name === "search" ? "Search" : "Snippet families"}
                </button>
              ))}
            </div>
            {healthError ? (
              <Offline message={healthError} onRetry={refreshHealth} />
            ) : health && !health.index_available ? (
              <Notice title="No index yet">
                Build one with <code className="text-desert">axiom index &lt;repo&gt;</code>, then reload.
              </Notice>
            ) : tab === "search" ? (
              <SearchPanel version={version} allVersions={allVersions} topK={topK} queryType={queryType} />
            ) : (
              <FamiliesPanel version={allVersions ? "" : version} />
            )}
          </main>
        </div>
      </div>
    </>
  );
}

const RESULT_MIN = 1;
const RESULT_MAX = 25;
const RESULT_TICKS = [1, 5, 10, 15, 20, 25];

/** The result-count slider: hairline track, sun fill, value shown large beside the label. */
function ResultCount({ value, onChange }: { value: number; onChange: (next: number) => void }) {
  const fill = ((value - RESULT_MIN) / (RESULT_MAX - RESULT_MIN)) * 100;
  return (
    <div>
      <div className="mb-3 flex items-baseline justify-between">
        <label htmlFor="result-count" className="text-mono-s uppercase opacity-70">
          Results
        </label>
        <output
          htmlFor="result-count"
          className="font-heading text-sun text-[1.75rem] leading-none font-light tabular-nums"
        >
          {String(value).padStart(2, "0")}
        </output>
      </div>
      <input
        id="result-count"
        type="range"
        min={RESULT_MIN}
        max={RESULT_MAX}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="range-sun w-full"
        style={{ "--fill": `${fill}%` } as React.CSSProperties}
      />
      <div className="relative mt-2 h-4" aria-hidden="true">
        {RESULT_TICKS.map((tick) => {
          const at = ((tick - RESULT_MIN) / (RESULT_MAX - RESULT_MIN)) * 100;
          return (
            <span
              key={tick}
              className={cx(
                "text-mono-s absolute -translate-x-1/2 tabular-nums transition-opacity",
                tick <= value ? "opacity-80" : "opacity-45",
              )}
              style={{ left: `calc(${at}% + ${7 - (at / 100) * 14}px)` }}
            >
              {tick}
            </span>
          );
        })}
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="text-mono-s mb-2 block uppercase opacity-60">{label}</span>
      {children}
    </label>
  );
}

function StatusPill({ health, error }: { health: Health | null; error: string | null }) {
  const ok = health && !error;
  return (
    <span className="text-mono-s flex items-center gap-2 rounded-full border border-white/12 px-3 py-1.5 uppercase">
      <span className={cx("size-1.5 rounded-full", ok ? "bg-oasis" : error ? "bg-sun" : "bg-stroke-2")} />
      {ok ? "API online" : error ? "API offline" : "Connecting"}
    </span>
  );
}

export function Notice({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <Card className="border-white/12 p-8 [--frame-dot:var(--color-stroke-3)]">
      <h2 className="text-heading-24">{title}</h2>
      <p className="text-body-16 mt-3 opacity-70">{children}</p>
    </Card>
  );
}

function Offline({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <Notice title="The API is offline">
      {message} Run <code className="text-desert">axiom serve --index-root &lt;dir&gt;</code> in the project, then{" "}
      <button onClick={onRetry} className="text-sun cursor-pointer underline underline-offset-4">
        retry
      </button>
      .
    </Notice>
  );
}

function SearchPanel({
  version,
  allVersions,
  topK,
  queryType,
}: {
  version: string;
  allVersions: boolean;
  topK: number;
  queryType: "" | QueryType;
}) {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const run = async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed) return;
    setQuery(trimmed);
    setLoading(true);
    setError(null);
    try {
      setResponse(
        await api.query({
          query: trimmed,
          top_k: topK,
          version: allVersions ? null : version || null,
          all_versions: allVersions,
          query_type: queryType || null,
        }),
      );
    } catch (caught) {
      setResponse(null);
      setError(caught instanceof ApiError ? caught.message : "The query failed.");
    } finally {
      setLoading(false);
    }
  };

  const onSubmit = (event: FormEvent) => {
    event.preventDefault();
    run(query);
  };

  return (
    <div>
      <form
        onSubmit={onSubmit}
        className="frame-dots border border-white/12 bg-[#120e0b] [--frame-dot:var(--color-stroke-3)]"
      >
        <label htmlFor="query" className="sr-only">
          Question about the code
        </label>
        <textarea
          id="query"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              run(query);
            }
          }}
          rows={2}
          placeholder="Where does this already happen?"
          className="font-heading placeholder:text-bone/45 block w-full resize-none bg-transparent px-5 pt-5 text-[clamp(1.25rem,2.2vw,1.625rem)] font-light outline-none md:px-6"
        />
        <div className="flex flex-wrap items-center justify-between gap-3 px-5 pt-2 pb-4 md:px-6">
          <div className="flex flex-wrap gap-2">
            {EXAMPLES.map((example) => (
              <button
                key={example.label}
                type="button"
                onClick={() => run(example.query)}
                className="text-mono-s cursor-pointer rounded-full border border-white/12 px-3 py-1.5 uppercase opacity-70 transition hover:border-white/30 hover:opacity-100"
              >
                {example.label}
              </button>
            ))}
          </div>
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="bg-bone text-night text-body-14 inline-flex cursor-pointer items-center gap-2 rounded-full px-5 py-2 font-medium transition hover:bg-white disabled:cursor-not-allowed disabled:opacity-55"
          >
            {loading ? "Searching…" : "Search"}
          </button>
        </div>
      </form>

      {error && <p className="text-body-14 text-sun mt-6">{error}</p>}
      {loading && <ResultsSkeleton />}
      {!loading && response && <ResponseView response={response} />}
      {!loading && !response && !error && (
        <p className="text-body-16 mt-10 max-w-[36rem] opacity-60">
          Try one of the three archetypes above. Name functions exactly for structural questions: “before{" "}
          <span className="font-mono text-[0.9em]">resolveTool</span>”, not “before dispatch”.
        </p>
      )}
    </div>
  );
}

function ResultsSkeleton() {
  return (
    <div className="mt-8 space-y-4" aria-busy="true" aria-label="Searching">
      {[0, 1, 2].map((index) => (
        <div key={index} className="h-40 animate-pulse rounded-sm border border-white/8 bg-white/[0.03]" />
      ))}
    </div>
  );
}

function ResponseView({ response }: { response: QueryResponse }) {
  const plan = response.query_plan;
  const familyByChunk = useMemo(() => {
    const map = new Map<string, { versions: string[]; size: number }>();
    for (const family of response.families ?? []) {
      for (const member of family.members)
        map.set(member.chunk_id, { versions: family.versions, size: family.members.length });
    }
    return map;
  }, [response.families]);

  return (
    <div className="mt-8">
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
        <span className="text-mono-s bg-sun/15 text-sun rounded-full px-3 py-1 uppercase">{plan.query_type}</span>
        <Meta>
          {response.passes_used} agent pass{response.passes_used === 1 ? "" : "es"}
        </Meta>
        <Meta>stopped: {response.stop_reason.replaceAll("_", " ")}</Meta>
        <Meta>{Math.round(response.elapsed_ms)} ms</Meta>
        <Meta>
          {response.results.length} result{response.results.length === 1 ? "" : "s"} · {response.version_ids.join(", ")}
        </Meta>
      </div>

      <div className="mt-6 grid gap-4 md:grid-cols-2">
        <Panel title="Fusion weights">
          <WeightBar weights={plan.strategy_weights} />
          <p className="text-body-14 mt-3 opacity-60">
            Score field: <span className="font-mono">{response.score_field}</span>
            {response.score_field === "rrf_score" && " (the cross-encoder did not run)"}
          </p>
        </Panel>
        <Panel title="Query plan">
          <ChipRow label="Identifiers" items={plan.extracted_identifiers} mono />
          <ChipRow label="Expanded with" items={plan.expansion_terms} />
          <ChipRow label="Sub-queries" items={plan.sub_queries} />
        </Panel>
      </div>

      <Timings timings={response.timings} total={response.elapsed_ms} />

      {response.degradations.length > 0 && (
        <details className="text-body-14 mt-4 rounded-sm border border-white/10 px-4 py-3 opacity-80">
          <summary className="cursor-pointer">
            {response.degradations.length} degradation{response.degradations.length === 1 ? "" : "s"} (the pipeline fell
            back gracefully)
          </summary>
          <ul className="mt-2 list-disc space-y-1 pl-5 font-mono text-[0.8rem] opacity-70">
            {response.degradations.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </details>
      )}

      {response.results.length === 0 ? (
        <Notice title="Nothing matched">Try naming a function, or search every version.</Notice>
      ) : (
        <ol className="mt-8 space-y-5">
          {response.results.map((result, index) => (
            <ResultCard
              key={`${result.chunk.metadata.version_id}-${result.chunk.chunk_id}-${index}`}
              result={result}
              rank={index + 1}
              family={familyByChunk.get(result.chunk.chunk_id)}
            />
          ))}
        </ol>
      )}
    </div>
  );
}

function Meta({ children }: { children: React.ReactNode }) {
  return <span className="text-mono-s uppercase opacity-60">{children}</span>;
}

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-sm border border-white/10 bg-white/[0.02] p-5">
      <h2 className="text-mono-s mb-4 uppercase opacity-60">{title}</h2>
      {children}
    </div>
  );
}

function WeightBar({ weights }: { weights: Partial<Record<SignalName, number>> }) {
  const shares = weightShares(weights);
  return (
    <div>
      <div className="flex h-2 overflow-hidden rounded-full bg-white/5">
        {shares.map(({ signal, percent }) => (
          <span key={signal} style={{ width: `${percent}%`, background: SIGNAL_COLOR[signal] }} />
        ))}
      </div>
      <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1">
        {shares.map(({ signal, percent }) => (
          <span key={signal} className="text-mono-s flex items-center gap-2 uppercase opacity-80">
            <span className="size-1.5 rounded-full" style={{ background: SIGNAL_COLOR[signal] }} />
            {SIGNAL_LABEL[signal]} {Math.round(percent)}%
          </span>
        ))}
      </div>
    </div>
  );
}

function ChipRow({ label, items, mono }: { label: string; items: string[]; mono?: boolean }) {
  if (!items.length) return null;
  return (
    <div className="mb-3 last:mb-0">
      <p className="text-body-14 mb-1.5 opacity-60">{label}</p>
      <div className="flex flex-wrap gap-1.5">
        {items.map((item) => (
          <span
            key={item}
            className={cx(
              "rounded-full border border-white/10 px-2.5 py-0.5 text-[0.8rem]",
              mono && "font-mono text-desert",
            )}
          >
            {item}
          </span>
        ))}
      </div>
    </div>
  );
}

function Timings({ timings, total }: { timings: Record<string, number>; total: number }) {
  const entries = Object.entries(timings).filter(([, ms]) => ms >= 1);
  if (!entries.length) return null;
  return (
    <div className="mt-4 rounded-sm border border-white/10 bg-white/[0.02] p-5">
      <h2 className="text-mono-s mb-4 uppercase opacity-60">Stage timings</h2>
      <div className="space-y-2">
        {entries.map(([stage, ms]) => (
          <div key={stage} className="grid grid-cols-[8.5rem_1fr_4.5rem] items-center gap-3">
            <span className="text-mono-s truncate opacity-70">{stage}</span>
            <span className="h-1.5 overflow-hidden rounded-full bg-white/5">
              <span className="bg-sun/70 block h-full" style={{ width: `${Math.min(100, (ms / total) * 100)}%` }} />
            </span>
            <span className="text-mono-s text-right opacity-70">{Math.round(ms)} ms</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function ResultCard({
  result,
  rank,
  family,
}: {
  result: Result;
  rank: number;
  family?: { versions: string[]; size: number };
}) {
  const { chunk } = result;
  const { location, metadata } = chunk;
  const reason = splitReason(result.match_reason);
  return (
    <li>
      <Card className="border-white/12 bg-[#120e0b] [--frame-dot:var(--color-stroke-3)]">
        <div className="flex flex-wrap items-start justify-between gap-4 border-b border-white/8 px-5 py-4 md:px-6">
          <div className="flex min-w-0 items-baseline gap-4">
            <span className="text-mono-m text-sun">{String(rank).padStart(2, "0")}</span>
            <div className="min-w-0">
              <p className="text-mono-m break-all">{locationRef(location)}</p>
              <p className="text-body-14 mt-1 opacity-60">
                {metadata.symbol ? <span className="text-desert font-mono">{metadata.symbol}</span> : "anonymous"} ·{" "}
                {metadata.kind} · {metadata.version_id}
                {family && family.versions.length > 1 && <> · family across {family.versions.join(", ")}</>}
              </p>
            </div>
          </div>
          <div className="flex flex-col items-end gap-2">
            <span className="text-mono-s opacity-60">score {result.score.toFixed(4)}</span>
            <div className="flex flex-wrap justify-end gap-1.5">
              {SIGNALS.filter((signal) => result.signals[signal] !== undefined).map((signal) => (
                <span
                  key={signal}
                  className="text-mono-s flex items-center gap-1.5 rounded-full border border-white/12 px-2.5 py-1"
                >
                  <span className="size-1.5 rounded-full" style={{ background: SIGNAL_COLOR[signal] }} />
                  {SIGNAL_LABEL[signal]} #{result.signals[signal]}
                </span>
              ))}
            </div>
          </div>
        </div>
        <p className="text-body-14 px-5 pt-4 opacity-80 md:px-6">
          <span className="text-mono-s mr-2 uppercase opacity-60">why {reason.signal}</span>
          {reason.detail}
        </p>
        <CodeBlock text={chunk.text} startLine={location.start_line} />
        <div className="flex justify-end px-5 pb-4 md:px-6">
          <Link
            href={`/chunk/${chunk.chunk_id}?version=${encodeURIComponent(metadata.version_id)}`}
            className="text-mono-s text-sun uppercase transition-opacity hover:opacity-80"
          >
            Open chunk details →
          </Link>
        </div>
      </Card>
    </li>
  );
}

export function CodeBlock({ text, startLine, maxLines = 18 }: { text: string; startLine: number; maxLines?: number }) {
  const lines = codeLines(text);
  const shown = lines.slice(0, maxLines);
  return (
    <div className="px-5 pt-3 pb-5 md:px-6">
      <pre className="text-mono-m overflow-x-auto rounded-sm bg-black/40 py-3 leading-6">
        {shown.map((line, index) => (
          <div key={index} className="grid grid-cols-[3.25rem_1fr]">
            <span className="pr-4 text-right opacity-55 select-none">{startLine + index}</span>
            <span className="pr-4 whitespace-pre">{line || " "}</span>
          </div>
        ))}
        {lines.length > shown.length && (
          <div className="text-mono-s pt-1 pl-[3.25rem] opacity-55">… {lines.length - shown.length} more lines</div>
        )}
      </pre>
    </div>
  );
}

function FamiliesPanel({ version }: { version: string }) {
  // Keyed by the version it was loaded for, so a version change reads as loading
  // without resetting state inside the effect.
  const [loaded, setLoaded] = useState<{ version: string; families: Family[]; total: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [filter, setFilter] = useState("");

  useEffect(() => {
    let cancelled = false;
    api
      .families({ limit: 200, multiOnly: true, diffs: true, version: version || undefined })
      .then((data) => {
        if (cancelled) return;
        setLoaded({ version, families: data.families, total: data.total });
        setError(null);
        const changed = data.families.find((family) => family.diffs.some(Boolean));
        setSelected((changed ?? data.families[0])?.family_id ?? null);
      })
      .catch(
        (caught) => !cancelled && setError(caught instanceof ApiError ? caught.message : "Could not load families."),
      );
    return () => {
      cancelled = true;
    };
  }, [version]);

  const families = loaded?.version === version ? loaded.families : null;

  const visible = useMemo(
    () =>
      (families ?? []).filter((family) => {
        const needle = filter.trim().toLowerCase();
        if (!needle) return true;
        const { symbol } = family.representative.metadata;
        return `${symbol ?? ""} ${family.representative.location.file_path}`.toLowerCase().includes(needle);
      }),
    [families, filter],
  );
  const current = visible.find((family) => family.family_id === selected) ?? visible[0];

  if (error) return <Notice title="Families unavailable">{error}</Notice>;
  if (!families) return <ResultsSkeleton />;
  if (!families.length)
    return <Notice title="One version only">Index a second version with axiom reindex to see families.</Notice>;

  return (
    <div>
      <p className="text-body-16 max-w-[40rem] opacity-70">
        The same function across versions, collapsed into one family. {families.length} families span more than one
        version; {families.filter((family) => family.diffs.some(Boolean)).length} of them changed along the way.
      </p>
      <div className="mt-6 grid gap-5 lg:grid-cols-[18rem_1fr]">
        <div>
          <input
            value={filter}
            onChange={(event) => setFilter(event.target.value)}
            placeholder="Filter by symbol or path"
            className="text-body-14 w-full rounded-md border border-white/12 bg-[#15110e] px-3 py-2"
          />
          <ul className="mt-3 max-h-[36rem] space-y-1 overflow-y-auto pr-1">
            {visible.map((family) => {
              const changed = family.diffs.some(Boolean);
              return (
                <li key={family.family_id}>
                  <button
                    onClick={() => setSelected(family.family_id)}
                    className={cx(
                      "w-full cursor-pointer rounded-sm px-3 py-2 text-left transition-colors",
                      current?.family_id === family.family_id ? "bg-white/[0.07]" : "hover:bg-white/[0.04]",
                    )}
                  >
                    <span className="text-mono-m flex items-center gap-2">
                      <span className={cx("size-1.5 shrink-0 rounded-full", changed ? "bg-sun" : "bg-stroke-3")} />
                      <span className="truncate">{family.representative.metadata.symbol ?? "block"}</span>
                    </span>
                    <span className="text-body-14 block truncate pl-3.5 opacity-60">
                      {family.representative.location.file_path}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        </div>
        {current && <FamilyDetail family={current} />}
      </div>
    </div>
  );
}

function FamilyDetail({ family }: { family: Family }) {
  // Members and diffs both arrive newest first; each diff names its two versions in its header.
  const oldestFirst = [...family.members].reverse();
  return (
    <Card className="border-white/12 min-w-0 bg-[#120e0b] [--frame-dot:var(--color-stroke-3)]">
      <div className="border-b border-white/8 px-5 py-4 md:px-6">
        <p className="text-mono-m text-desert">{family.representative.metadata.symbol ?? "block"}</p>
        <p className="text-body-14 mt-1 opacity-60">
          {family.representative.location.file_path} · stability {family.stability.toFixed(2)}
        </p>
        <div className="mt-4 flex flex-wrap items-center gap-2">
          {oldestFirst.map((member, index) => (
            <span
              key={`${member.metadata.version_id}-${member.chunk_id}`}
              className="text-mono-s flex items-center gap-2 opacity-80"
            >
              {index > 0 && <span className="bg-stroke-3 h-px w-6" />}
              <span className="rounded-full border border-white/12 px-2.5 py-1">
                {member.metadata.version_id} · L{member.location.start_line}-{member.location.end_line}
              </span>
            </span>
          ))}
        </div>
      </div>
      <div className="space-y-4 px-5 py-5 md:px-6">
        {family.diffs.every((diff) => !diff) ? (
          <p className="text-body-14 opacity-60">Identical in every version: stored once, reused everywhere.</p>
        ) : (
          family.diffs.map((diff, index) => (diff ? <DiffView key={index} diff={diff} /> : null))
        )}
        <details>
          <summary className="text-mono-s cursor-pointer uppercase opacity-60">Latest source</summary>
          <CodeBlock text={family.representative.text} startLine={family.representative.location.start_line} />
        </details>
      </div>
    </Card>
  );
}

function DiffView({ diff }: { diff: string }) {
  const lines = codeLines(diff);
  return (
    <pre className="text-mono-m overflow-x-auto rounded-sm bg-black/40 py-3 leading-6">
      {lines.map((line, index) => {
        const kind = diffLineKind(line);
        return (
          <div
            key={index}
            className={cx(
              "px-4 whitespace-pre",
              kind === "header" && "opacity-55",
              kind === "added" && "bg-oasis/15 text-[#9be0cf]",
              kind === "removed" && "bg-sun/10 text-desert",
            )}
          >
            {line || " "}
          </div>
        );
      })}
    </pre>
  );
}
