// Typed client for the Axiom REST API (docs/API.md). Paths go through the
// /api/axiom rewrite in next.config.ts.

export type SignalName = "dense" | "sparse" | "structural";
export type QueryType = "semantic" | "structural" | "usage" | "hybrid";

export interface ChunkLocation {
  file_path: string;
  start_line: number;
  end_line: number;
}

export interface ChunkMetadata {
  symbol: string | null;
  kind: string;
  parent_symbol: string | null;
  is_exported: boolean;
  imports: string[];
  calls: string[];
  docstring: string | null;
  language: string;
  version_id: string;
  commit_sha: string | null;
}

export interface Chunk {
  chunk_id: string;
  content_hash: string;
  text: string;
  location: ChunkLocation;
  metadata: ChunkMetadata;
}

export interface Result {
  chunk: Chunk;
  score: number;
  match_reason: string;
  signals: Partial<Record<SignalName, number>>;
  optimization_hint: string | null;
}

export interface QueryPlan {
  original_query: string;
  query_type: QueryType;
  sub_queries: string[];
  extracted_identifiers: string[];
  expansion_terms: string[];
  strategy_weights: Partial<Record<SignalName, number>>;
}

export interface FamilyMemberRef {
  chunk_id: string;
  version_id: string;
  location: string;
  score: number;
}

export interface QueryFamily {
  family_id: string;
  representative: string;
  versions: string[];
  members: FamilyMemberRef[];
}

export interface QueryResponse {
  results: Result[];
  query_plan: QueryPlan;
  elapsed_ms: number;
  passes_used: number;
  timings: Record<string, number>;
  warnings: string[];
  profile: string;
  stop_reason: string;
  score_field: string;
  version_ids: string[];
  degradations: string[];
  families: QueryFamily[];
}

export interface QueryRequest {
  query: string;
  top_k?: number;
  version?: string | null;
  all_versions?: boolean;
  query_type?: QueryType | null;
}

export interface VersionInfo {
  version_id: string;
  created_at: string;
  chunk_count: number;
  parent_version: string | null;
}

export interface VersionsResponse {
  active_version: string | null;
  versions: VersionInfo[];
}

export interface Family {
  family_id: string;
  representative: Chunk;
  members: Chunk[];
  versions: string[];
  stability: number;
  /** One unified diff per consecutive version pair; "" when nothing changed. */
  diffs: string[];
}

export interface FamiliesResponse {
  families: Family[];
  total: number;
  version_ids: string[];
}

export interface Health {
  status: string;
  index_available: boolean;
  profile: string;
  version: string;
}

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public code?: string,
  ) {
    super(message);
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/api/axiom${path}`, {
      ...init,
      headers: { "content-type": "application/json", ...(init?.headers ?? {}) },
      cache: "no-store",
    });
  } catch {
    throw new ApiError("The Axiom API is not reachable. Start it with `axiom serve`.", 0);
  }
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = body?.detail ?? `HTTP ${response.status}`;
    throw new ApiError(
      response.status >= 500 && !body ? "The Axiom API is not reachable. Start it with `axiom serve`." : String(detail),
      response.status,
      body?.error,
    );
  }
  return body as T;
}

export const api = {
  health: () => call<Health>("/health"),
  versions: () => call<VersionsResponse>("/versions"),
  query: (request: QueryRequest) => call<QueryResponse>("/query", { method: "POST", body: JSON.stringify(request) }),
  families: (params: { limit?: number; multiOnly?: boolean; diffs?: boolean; version?: string }) => {
    const search = new URLSearchParams();
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.multiOnly) search.set("multi_only", "true");
    if (params.diffs) search.set("diffs", "true");
    if (params.version) search.set("version", params.version);
    return call<FamiliesResponse>(`/families?${search}`);
  },
  chunk: (chunkId: string, version?: string) =>
    call<Chunk>(`/chunk/${encodeURIComponent(chunkId)}${version ? `?version=${encodeURIComponent(version)}` : ""}`),
};

export const SIGNAL_LABEL: Record<SignalName, string> = {
  dense: "Dense",
  sparse: "BM25",
  structural: "Call graph",
};

export const SIGNAL_COLOR: Record<SignalName, string> = {
  dense: "var(--color-dawn)",
  sparse: "var(--color-sun)",
  structural: "var(--color-oasis)",
};
