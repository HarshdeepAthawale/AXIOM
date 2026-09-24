# API

HTTP endpoint contracts, error codes, CLI reference.

**Owner:** Harshdeep Athawale
**Last updated:** 2026-09-23
**Status:** Draft

Related: [PRD.md](PRD.md) · [Schema.md](Schema.md) · [Rules.md](Rules.md#92-error-taxonomy) · [Setup.md](Setup.md#7-environment-variables) · [TestPlan.md](TestPlan.md) · [NonGoals.md](NonGoals.md) · [Security.md](Security.md) · [Deployment.md](Deployment.md) · [OpenQuestions.md](OpenQuestions.md#oq-04--project-name-collision-with-a-rival-submission) · [Tracker.md](Tracker.md)

---

## 1. Scope and Authority

This document is the wire-format and CLI-surface contract for Axiom: every HTTP request/response
shape, every HTTP status code, every machine-readable error code, and every CLI subcommand,
flag, and exit code. It does not redefine the shared data model — `RetrievalResult`, `QueryPlan`,
`Chunk`, `VersionManifest` and the rest are owned by [Schema.md](Schema.md) and this document
references them by name, never restates their fields.

Two surfaces exist, both `FR-24`/`FR-23` in [PRD.md](PRD.md#5-functional-requirements), both typed
by the same Pydantic models:

| Surface | Transport | Port | Section |
|---|---|---|---|
| HTTP API | FastAPI + uvicorn | `8000` (`AXIOM_API_PORT`) | §2–§6 |
| CLI | `typer` | — | §7–§8 |

**The HTTP API is dev-only and unauthenticated.** This is not an oversight scheduled for a later
middleware pass — it is a deployment posture, locked as
[`NG-08`](NonGoals.md#ng-08--no-authentication-authorisation-or-multi-tenancy): no login, no API
key, no JWT, no RBAC, no rate limiter. Anyone who can reach port 8000 can read the index, and
anyone who can read the index can already read the repo it was built from, so the API grants no
privilege the filesystem did not already grant. There is no production SLA
([`NG-10`](NonGoals.md#ng-10--no-production-sla-uptime-or-ha-guarantee)) and nothing is hosted
([`NG-13`](NonGoals.md#ng-13--no-hosted-cloud-deployment)) — the API binds loopback by default and
is meant to be run by the person using it, on their own machine, for a demo or for local
automation.

**The canonical paths are `/v1/`-prefixed, and this document is where that is settled.** The suite
previously carried both forms — prefixed in this document and in `TC-070`, unprefixed in
[PRD.md](PRD.md) `FR-24` and TechSpecifications §4.11 — with no statement of which won. `/v1/` wins:
it is what the tests assert and what `src/axiom/api/routes.py` registers as the documented surface.
`FR-24` and TechSpecifications §4.11 must quote this section rather than restate a second path set.

`src/axiom/api/routes.py` also registers each endpoint a second time at the **bare** path
(`POST /query`, `GET /versions`, …) as a back-compatibility alias for anything written against the
unprefixed form. The aliases are marked `include_in_schema=False`, so `/openapi.json` and the
generated docs advertise exactly the `/v1/` set in §3 and nothing else. **Do not document, test, or
demo the bare paths** — they exist so a stale client gets an answer instead of a 404, not as a
second supported surface.

**Why a `/v1/` prefix at all, given there is no versioning policy:** see §9. In short,
[`NG-11`](NonGoals.md#ng-11--no-ide-plugin) puts an IDE plugin out of scope for this cycle but names
this exact contract as the thing a future plugin would consume as "a thin client... of a frozen
contract rather than a rewrite." The prefix exists for that hypothetical future consumer; it does
not imply a `/v2/` is planned or that this doc promises compatibility across changes.

---

## 2. Base URL and Transport

| Setting | Env var | Default | Notes |
|---|---|---|---|
| Bind host | `AXIOM_API_HOST` | `127.0.0.1` | `0.0.0.0` inside Docker only, per [Deployment.md](Deployment.md) |
| Bind port | `AXIOM_API_PORT` | `8000` | [Setup.md §7.6](Setup.md#76-services) |
| Allowed browser origins | `AXIOM_API_CORS_ORIGINS` | *(unset)* | Comma-separated exact origins, or `*`. Unset means no CORS middleware at all |
| Client base URL (used by the Streamlit UI) | `AXIOM_API_BASE_URL` | `http://127.0.0.1:8000` | [Setup.md §7.6](Setup.md#76-services) |

The full environment-variable reference is owned by [Setup.md §7](Setup.md#7-environment-variables);
this document names only the variables that directly shape the request/response contract.

Start the server with `axiom serve` (see §7). Every request and response body is `application/json`,
UTF-8, no trailing newline requirement.

**CORS is off unless you ask for it.** `AXIOM_API_CORS_ORIGINS` takes a comma-separated list of exact
browser origins (`http://localhost:5173,http://127.0.0.1:5173`), or `*`. Unset — the default — no CORS
middleware is installed at all and the surface behaves exactly as [Security.md](Security.md) describes
it. This exists for one reason: a browser frontend served from its own dev-server port is
cross-origin, and no amount of correct backend code makes the browser send that request without it.
It is an opt-in rather than a default because this surface is unauthenticated per §1, so a permissive
CORS header is the difference between "a local dev server" and "any page the user has open can read
their source index".

When it is on: `allow_methods` is `GET, POST, OPTIONS` and `allow_headers` is `content-type` — the
methods and header the five endpoints actually use, not `*`. `allow_credentials` is always `false`,
in both the explicit-origin and `*` modes; there is no cookie or session on this surface for a
browser to attach. A `*` value is honoured for a demo whose frontend port is not stable, and logs a
`WARNING` when it is used.

---

## 3. Endpoints

Five endpoints implement `FR-24`. All four are typed by Pydantic models: the payload models are the
shared [Schema.md](Schema.md) contract (`RetrievalResult`, `QueryPlan`, `Chunk`, `VersionManifest`
summaries); the request/response *envelopes* around them (e.g. the object that wraps `results` and
`query_plan` together with `elapsed_ms`) are API-specific models in `src/axiom/api/models.py`, not
part of `src/axiom/schema/` — envelopes are allowed to add HTTP-only bookkeeping fields without
touching the shared contract four people code against.

### 3.1 `POST /v1/query`

Runs the full retrieval pipeline — classify, plan, retrieve (dense/sparse/structural), fuse,
rerank, optionally refine — and returns ranked results. This is the endpoint
[TestPlan.md TC-070](TestPlan.md) and [TC-071](TestPlan.md) exercise directly; the shapes below are
built to satisfy both exactly.

**Request body**

| Field | Type | Required | Default | Constraint |
|---|---|---|---|---|
| `query` | `str` | yes | — | `min_length=1` after stripping; empty or whitespace-only is a validation failure, not a pipeline degradation (contrast [Rules.md §3](Rules.md#rule-3--never-raise-on-bad-input-degrade)'s internal ladder, which handles a query that goes empty only *after* passing this gate, e.g. via truncation) |
| `top_k` | `int` | no | `AXIOM_TOP_K_DEFAULT` (`10`) | `1 <= top_k <= AXIOM_TOP_K_MAX` (`100`), both from [Setup.md §7.4](Setup.md#74-retrieval-and-fusion) |
| `version` | `str \| null` | no | `null` (registry's active version) | pattern `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$` — no `/`, no `..`, no leading `.`; rejects path-traversal-shaped input at the validation layer, before any filesystem access is attempted |
| `all_versions` | `bool` | no | `false` | when `true`, searches every indexed version and collapses near-duplicates into `SnippetFamily`-derived results (`FR-21`); mutually informative with, not exclusive of, `version` — `version` is ignored when `all_versions=true` |
| `query_type` | `QueryType \| null` | no | `null` (auto-classify) | one of `semantic`, `structural`, `usage`, `hybrid` ([Schema.md §3.1](Schema.md#31-querytype-semantics)); forces the classifier's output instead of running it |

Example:

```json
{
  "query": "Which files call tool XYZ before tool ABC?",
  "top_k": 10
}
```

**Response body — `200 OK`**

Twelve fields, all of them always present. `api/models.py`'s `QueryResponse` is `extra="forbid"`, so
this table is exhaustive in both directions: nothing else appears, and none of these is optional.

The first six are the core contract:

| Field | Type | Description |
|---|---|---|
| `results` | `list[`[`RetrievalResult`](Schema.md#9-retrievalresult)`]` | `len(results) <= top_k`. Note `RetrievalResult` carries `score`, not a separate `rerank_score` — which of the two scales `score` is on is said by `score_field` below |
| `query_plan` | [`QueryPlan`](Schema.md#10-queryplan) | the plan from the final agent pass |
| `elapsed_ms` | `float` | `>= 0`; wall-clock for the whole request |
| `passes_used` | `int` | `0..AXIOM_AGENT_MAX_PASSES`. **The bound counts TOTAL retrieve→fuse→hydrate→rerank cycles, initial pass included** — `AXIOM_AGENT_MAX_PASSES=2` means one initial pass plus at most one refinement, never three. `0` occurs only when the query went empty after normalisation and no pass ran; `stop_reason` says so (`FR-13`, `US-5`) |
| `timings` | `dict[str, float]` | per-stage elapsed ms, keyed by the closed stage vocabulary in [Rules.md §9.1](Rules.md#91-logging-discipline) — on the query path: `plan`, `agent.fan_out`, `fuse`, `hydrate`, `rerank`, `format`, and `evolutionary` when enabled. A stage that runs twice (the loop's second pass) is **summed** into one key. Satisfies `NFR-10` |
| `warnings` | `list[object]` | degradations that did not fail the request; see §5. Each item is `{"code": str, "detail": str}`. Empty list on a clean run |

The remaining six are the HTTP-only bookkeeping §3 permits an envelope to carry. Each earns its
place in the UI and in `--json` diffs, and a client that ignores them still parses correctly:

| Field | Type | Description |
|---|---|---|
| `profile` | `str` | the config profile that answered — `default`, `demo`, `eval`, `fast` or `accurate` |
| `stop_reason` | `str` | why the agent loop stopped: sufficiency met, pass bound reached, budget expired, the planner's rewrite matched the query already run, or the query was empty |
| `score_field` | `str` | **which field `RetrievalResult.score` carries**: `"rerank_score"` when the cross-encoder ran, `"rrf_score"` on the passthrough rung. The two differ by roughly two orders of magnitude (RRF sits near `1/(60+rank)`), so a client that renders `score` without reading this is how a demo accidentally overclaims |
| `version_ids` | `list[str]` | the index versions actually searched, after `all_versions` resolution |
| `degradations` | `list[str]` | every ladder rung taken during this query, in order |
| `families` | `list[object]` | cross-version groups, one per result, in result order — **empty unless the request set `all_versions`**. Present either way: a key that appears only on one code path forces every client to branch on its absence. Each item is `{"family_id": str, "representative": str, "versions": list[str], "members": [{"chunk_id", "version_id", "location", "score"}]}`. `representative` is the `chunk_id` of the member that survived into `results`; `members` keeps the collapsed rows reachable ([TC-085](TestPlan.md)). Deliberately **no `stability`** — see the note below |

`families` carries no stability number, and this is not an omission. [`SnippetFamily.stability`](Schema.md#11-snippetfamily)
is "in how many of the indexed versions does this snippet exist", computed over the whole corpus. The
only thing derivable from a result list is how many versions survived into the top-k, which for an
unchanged snippet is always exactly one: `chunk_id` is `digest(text, file_path, start_line)` and carries
no version, so two versions' copies of an untouched function are literally the same candidate and fusion
has already merged them. Publishing that ratio as "stability" would put `0.50` on a query card for the
same family `GET /v1/families` reports as `1.00`. Ask §3.3 for stability.

**The determinism contract excludes `elapsed_ms` *and every value under `timings`.*** Both are
wall-clock readings and neither is reproducible across runs; comparing `timings` **key sets** is
reproducible and is what a determinism assertion should compare. Every other field, including
`stop_reason`, `score_field`, `version_ids` and `degradations`, is byte-identical across repeat runs
at the same seed. [TestPlan.md TC-066](TestPlan.md) must exclude both, not `elapsed_ms` alone — with
only `elapsed_ms` excluded it fails on its first run and on every run after.

Example:

```json
{
  "results": [
    {
      "chunk": { "chunk_id": "9f2c41d80ba7e35617c4d9a0e8b3f512", "...": "see Schema.md §6" },
      "score": 0.8956,
      "match_reason": "structural: handleDeeplink calls preprocessInput before resolveTool (rank 2 in call-graph)",
      "signals": { "dense": 3, "sparse": 11, "structural": 2 },
      "optimization_hint": null
    }
  ],
  "query_plan": {
    "original_query": "Which files call tool XYZ before tool ABC?",
    "query_type": "structural",
    "sub_queries": [],
    "extracted_identifiers": ["XYZ", "ABC"],
    "expansion_terms": [],
    "strategy_weights": { "dense": 0.2, "sparse": 0.2, "structural": 0.6 }
  },
  "elapsed_ms": 742.3,
  "passes_used": 1,
  "timings": { "plan": 58.1, "agent.fan_out": 56.4, "fuse": 1.6, "hydrate": 12.0, "rerank": 601.0, "format": 8.9 },
  "warnings": [],
  "profile": "demo",
  "stop_reason": "sufficient",
  "score_field": "rerank_score",
  "version_ids": ["v2.3.1"],
  "degradations": []
}
```

`axiom query --all-versions --json` adds one further key, `families`, carrying the members a
collapsed row hides. It is additive and appears only under that flag.

**Status codes**

| Code | When |
|---|---|
| `200` | Always, for any well-formed, in-bounds request — including a query that degrades internally (empty result set, reranker passthrough, budget exhaustion). A degraded-but-valid answer is success, per [Rules.md Rule 3](Rules.md#rule-3--never-raise-on-bad-input-degrade). |
| `422` | Request fails validation: empty/whitespace `query`, `top_k` out of `[1, AXIOM_TOP_K_MAX]`, malformed `version` (including path-traversal-shaped values). See [TC-071](TestPlan.md). |
| `413` | Request body exceeds `AXIOM_API_MAX_BODY_BYTES` (default `262144`, 256 KiB — a real query is a few hundred bytes; the 1 MB adversarial body in [TC-071](TestPlan.md) is well past any legitimate use). This limit is API-specific and documented here rather than in [Setup.md](Setup.md)'s general environment-variable table because it governs request shape, not pipeline behaviour. |
| `503` | A required model is unavailable and cannot be fetched (`MODEL_UNAVAILABLE`, §6), or the index root itself is unreadable. |
| `404` | A `version` was named that does not resolve (`VERSION_NOT_FOUND`, §6). An otherwise-healthy index, wrongly scoped. |
| `500` | An internal contract violation (`AxiomContractError`) reached the API boundary. Should never happen in a shipped build; see §6. |

### 3.2 `GET /v1/versions`

Lists every indexed version's summary, sourced from `.axiom/registry.json`
([Schema.md §14.1](Schema.md#141-registryjson)) — the same fields the registry keeps duplicated
from each `VersionManifest` specifically so this listing never has to open every manifest file.

**Query parameters:** none.

**Response body — `200 OK`**

| Field | Type | Description |
|---|---|---|
| `active_version` | `str` | the version used by `POST /v1/query` when `version` is omitted |
| `versions` | `list[object]` | one entry per indexed version |

Each entry in `versions`:

| Field | Type | Description |
|---|---|---|
| `version_id` | `str` | matches [`VersionManifest.version_id`](Schema.md#12-versionmanifest) |
| `created_at` | `str` | ISO-8601 UTC |
| `chunk_count` | `int` | `>= 0` |
| `parent_version` | `str \| null` | derivation parent, or `null` for a cold build |

Example:

```json
{
  "active_version": "v2.3.1",
  "versions": [
    { "version_id": "v2.2.0", "created_at": "2026-09-16T09:12:40Z", "chunk_count": 9871, "parent_version": null },
    { "version_id": "v2.3.0", "created_at": "2026-09-17T11:44:05Z", "chunk_count": 10102, "parent_version": "v2.2.0" },
    { "version_id": "v2.3.1", "created_at": "2026-09-18T14:03:22Z", "chunk_count": 10248, "parent_version": "v2.3.0" }
  ]
}
```

**Status codes:** `200` always when the index root is readable; `503` if `.axiom/registry.json` is
missing or corrupt (see `INDEX_UNAVAILABLE` in §6).

### 3.3 `GET /v1/families`

Browses snippet families across the indexed versions — the *no-query* half of `FR-21`, and the HTTP
equivalent of `axiom families`. Where §3.1's `families` answers "what else did this one result have",
this answers "what has changed in this codebase across versions" without a query at all. Both call
`pipeline.list_families`, so the two surfaces cannot disagree about what the corpus contains.

**Query parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `version` | `str \| null` | `null` (every built version, newest first) | restrict to one version; same pattern constraint as §3.1's `version` |
| `limit` | `int` | `20` | page size, `0 <= limit <= 500`. `0` means "every match", still capped at `500` |
| `multi_only` | `bool` | `false` | only families spanning two or more versions |
| `diffs` | `bool` | `false` | attach per-transition unified diffs. **Costs real work** — a diff per transition for every family on the page — so request it for the one family a user expanded, not for the list |

**Response body — `200 OK`**

| Field | Type | Description |
|---|---|---|
| `families` | `list[`[`SnippetFamily`](Schema.md#11-snippetfamily)`]` | at most `limit` of them. Serialised fields are exactly `family_id`, `representative`, `versions`, `members`, `stability`, `diffs`. `is_multi_version` is a Python property, **not** a wire field — derive it as `versions.length >= 2` |
| `total` | `int` | families matching the filter **before** `limit` truncated — without it a client cannot tell "there are 12" from "there are 400 and you asked for 12" |
| `version_ids` | `list[str]` | the versions the families were computed over, newest first. This list's length is `stability`'s denominator |

Two `representative` fields, two different things — the one trap in this surface. Here it is the full
[`Chunk`](Schema.md#6-chunk) object (`representative.metadata.symbol`, `representative.location.file_path`);
in §3.1's `families` block it is a bare `chunk_id` string. §3.1 is a pointer into `results`, which already
carries the chunk; this endpoint has no `results` to point into, so it carries the chunk itself.

Grouping needs vectors: two same-named functions in one file are one family only if they are also
near-identical ([TC-082](TestPlan.md)). Those come from the content-addressed blob store, and when
numpy or the blobs are absent every family degrades to a single member and says so in the log — the
honest answer rather than a merge that never compared anything ([Rules.md §3](Rules.md#rule-3--never-raise-on-bad-input-degrade)).

**Status codes**

| Code | When |
|---|---|
| `200` | families were computed, including the empty-list case |
| `404` | a `version` was named that does not resolve (`VERSION_NOT_FOUND`, §6) |
| `422` | `version` is malformed, or `limit` is out of range |
| `503` | no index has been built at all (`INDEX_UNAVAILABLE`) |

### 3.4 `GET /v1/chunk/{chunk_id}`

Hydrates one full [`Chunk`](Schema.md#6-chunk) by id — text, location, metadata. Used by the
Streamlit UI and any client that received a `chunk_id` from a prior `/v1/query` call and wants the
full record again without re-running retrieval.

**Path parameter**

| Name | Type | Constraint |
|---|---|---|
| `chunk_id` | `str` | pattern `^[0-9a-f]{32}$` — a blake2b-128 hex digest, per [Rules.md Rule 1](Rules.md#rule-1--ids-are-sacred). Never mutated, never re-derived; looked up by exact string equality against `chunks.jsonl` for the resolved version. |

**Query parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `version` | `str \| null` | `null` (registry's active version) | scopes the lookup; same pattern constraint as §3.1's `version` field |

**Response body — `200 OK`:** the [`Chunk`](Schema.md#6-chunk) object, unwrapped (no envelope) —
`chunk_id`, `content_hash`, `text`, `location`, `metadata`.

**Status codes**

| Code | When |
|---|---|
| `200` | `chunk_id` resolves in the requested version |
| `404` | `chunk_id` is well-formed but not present in the resolved version's index (`CHUNK_NOT_FOUND`, §6), **or** a `version` was named that does not resolve (`VERSION_NOT_FOUND`) |
| `422` | `chunk_id` does not match the 32-hex-char pattern, or `version` is malformed |
| `500` | The matching row failed its own validation — a `chunk_id`/`content_hash` that disagrees with its body is corrupted state, not user input (`CONTRACT_VIOLATION`, §6) |
| `503` | No index resolves at all, with no `version` named (`INDEX_UNAVAILABLE`) |

The 404/503 split is whose fault it is: a version the *client* named that does not exist is a
scoping error against a healthy index; no index at all is the server having nothing to serve.

### 3.5 `GET /v1/health`

Liveness probe, plus the canonical way to force eager model loading before the demo — this is the
endpoint [Setup.md's troubleshooting table](Setup.md#9-troubleshooting) points at for the
"first query after startup takes 20-40 s" symptom, and the endpoint
[Deployment.md](Deployment.md)'s demo runbook warms before the jury ever sees a query.

**Query parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `warm` | `bool` | `false` | when `true`, eagerly constructs the lazy-loaded model singletons ([Rules.md AP-06](Rules.md#ap-06--loading-models-at-import-time)'s pattern: ONNX sessions for the embedder and reranker, GGUF mmap for the LLM if `AXIOM_LLM_ENABLED=true`) before responding, instead of leaving them to construct lazily on the first real query |

**Response body — `200 OK`**

| Field | Type | Description |
|---|---|---|
| `status` | `str` | always `"ok"` for a `200` |
| `warm` | `bool` | echoes the request; `true` means warming was requested and completed on this call |
| `warmed` | `list[str]` | which components were actually constructed **this call**, in load order (`embedder`, `reranker`, `llm` — the last only when `AXIOM_LLM_ENABLED=true`); empty when `warm=false` **or** when everything was already warm from a prior call. The field reports what this call built, not what is currently loaded, so a near-instant second `?warm=true` with an empty list is self-explanatory rather than a failure |
| `elapsed_ms` | `float` | request wall-clock; on the first `?warm=true` call this is the 20-40 s cold-load cost the troubleshooting table describes, on every subsequent call (warm or not) it is near-zero |
| `index_available` | `bool` | whether a built index resolved. **`false` is still a `200`** — the process is alive, there is simply nothing to search yet, and failing a container healthcheck for that would be reporting a non-fault |
| `profile` | `str` | the active config profile |
| `version` | `str` | the installed `axiom` package version |

Example — the canonical pre-demo warm call:

```bash
curl -s "http://127.0.0.1:8000/v1/health?warm=true"
```

```json
{
  "status": "ok",
  "warm": true,
  "warmed": ["embedder", "reranker", "llm"],
  "elapsed_ms": 24531.8,
  "index_available": true,
  "profile": "demo",
  "version": "0.1.0"
}
```

**Status codes:** `200` for a live process, warmed or not. `503` only if `warm=true` was requested
and a required model could not be loaded at all (`MODEL_UNAVAILABLE`, §6) — a plain `GET /v1/health`
with no `warm` parameter never touches a model and therefore never returns `503` for a model reason.

---

## 4. Request Validation and Limits

All numeric and pattern constraints in §3 are enforced by Pydantic field validators on the request
model, before any stage of the pipeline runs and before any file is opened. This is deliberate and
tested: [TC-071](TestPlan.md) asserts "no filesystem access attempted" for a path-traversal-shaped
`version` value, by patching `open` and confirming it is never called. Validation failure is
therefore always `422` (or `413` for body size), never a `500`, and never reaches
`AxiomContractError`/`IndexNotFoundError` territory.

| Limit | Source | Value |
|---|---|---|
| `top_k` default | `AXIOM_TOP_K_DEFAULT` | `10` ([Setup.md §7.4](Setup.md#74-retrieval-and-fusion)) |
| `top_k` ceiling | `AXIOM_TOP_K_MAX` | `100` ([Setup.md §7.4](Setup.md#74-retrieval-and-fusion)) |
| Request body ceiling | `AXIOM_API_MAX_BODY_BYTES` | `262144` (256 KiB), API-specific, not in [Setup.md](Setup.md)'s general table |
| `version` / `chunk_id` shape | field pattern | no path separators, no leading dot; rejects `../../etc`-style input outright |

---

## 5. Degradation vs. Error: What Surfaces Where

[Rules.md §3](Rules.md#rule-3--never-raise-on-bad-input-degrade) is unambiguous: a malformed,
empty, or pathologically long *query* must never terminate a run, and most stage failures degrade
along a declared ladder rather than raising. That principle has a direct HTTP consequence this
section states explicitly, because it is easy to get backwards: **degradation is a `200` with
information in the body; only a genuine contract violation, a missing index, or an unrecoverable
model failure is an HTTP error.**

| Condition | Surfaces as | Why |
|---|---|---|
| Reranker exceeds `AXIOM_RERANKER_TIMEOUT_MS` ([Setup.md §7.2](Setup.md#72-models)) | `200`, a `warnings` entry, RRF order returned unchanged, and **`score_field: "rrf_score"`** in the envelope. (`RetrievalResult` has no `rerank_score` field — `score_field` is how passthrough is detected on the wire; `rerank_score` is a `FusedResult` field and stays internal.) | [Rules.md](Rules.md)'s reranker degradation ladder: "cross-encode → truncate → pass RRF order through unchanged." A timeout is exactly this rung, not a failure. |
| Agent loop exhausts `AXIOM_AGENT_WALL_CLOCK_MS` ([Setup.md §7.3](Setup.md#73-agent-and-llm)) | `200`, `warnings` entry `{"code": "AGENT_BUDGET_EXCEEDED", "detail": "..."}`, best-results-so-far returned, `passes_used` reflects what actually ran | There is deliberately no budget *exception* class: the loop returns best-so-far rather than raising, so this is a warning by construction ([Rules.md §9.2](Rules.md#92-error-taxonomy)). |
| A `query_type` override arrives that this build's pipeline entry point cannot forward | `200`, `warnings` entry `{"code": "QUERY_TYPE_UNSUPPORTED", "detail": "..."}`, the classifier's own decision used instead | A request the server understood but could not honour exactly is still a successful answer; the client is told which of the two it got rather than silently given the other. |
| Any first-stage signal (dense/sparse/structural) returns nothing | `200`, no dedicated warning entry (visible instead via `signals` on individual results and via `timings`) | Declared ladder rung, e.g. sparse "tokeniser yields zero tokens → empty list." An empty signal is routine, not exceptional — see the `TC-055` empty-signal renormalisation test. |
| A model load fails but a fallback is cached and loadable | `200`, no error; the fallback model's identity is visible in `GET /v1/health?warm=true`'s `warmed` list and in server logs at `WARNING` | `NFR-07` mandatory degradation: every model has a declared fallback. |
| Every rung of a degradation ladder failed — `DegradationExhaustedError` (typically `AXIOM_OFFLINE=true` plus an uncached model) | `503`, `{"error": "MODEL_UNAVAILABLE", ...}` | There is no ladder rung left — see §6. This is the one model-loading condition Rule 3 does not ask to be swallowed, because [Setup.md §5.4](Setup.md#54-pre-download-for-an-offline-demo-do-this-the-day-before) deliberately wants it to fail fast and visibly rather than hang on a socket timeout in front of the jury. |
| `chunk_id` or `version` malformed, `top_k` out of range, empty `query`, oversized body | `422` / `413` | Input validation, not pipeline degradation — the request never reaches a stage. See §4. |
| An emitted id not in the corpus, a wrong embedding dimension, a non-contiguous rank | `500`, `{"error": "CONTRACT_VIOLATION", ...}` | [Rules.md §3](Rules.md#rule-3--never-raise-on-bad-input-degrade)'s one deliberately-fatal category: `AxiomContractError` "means the code is broken; failing loudly... is cheaper than a 0.0 score." Should never occur in a shipped build. |

---

## 6. Error Code Reference

This section resolves the forward reference in [Setup.md §5.4](Setup.md#54-pre-download-for-an-offline-demo-do-this-the-day-before):
*"`AXIOM_OFFLINE=true` makes any attempted network fetch raise `MODEL_UNAVAILABLE` immediately... see API.md §6."*

Every non-`2xx` response body has the same fixed shape, regardless of which layer produced it —
FastAPI's default validation-error body (a list of Pydantic error objects under `detail`) is
overridden by a custom exception handler so a client only ever parses one shape:

```json
{ "error": "<ERROR_CODE>", "detail": "<human-readable, one line, no stack trace>" }
```

[TC-071](TestPlan.md) asserts this directly: "no stack trace in the body" for every rejected
request. Full tracebacks are logged server-side at `DEBUG` per [Rules.md §9.1](Rules.md#91-logging-discipline)
and never serialised into a response.

The table below maps every exception in [Rules.md §9.2](Rules.md#92-error-taxonomy)'s error
taxonomy (`src/axiom/core/errors.py`) to its HTTP treatment. The class names are `Axiom*`, matching
both that module and [`ADR-015`](Decisions.md#adr-015--rename-prism-to-axiom) — an earlier draft of
this section used `Prism*`, which named classes the code does not define.

| Exception (`axiom.core.errors`) | HTTP status | `error` code | Notes |
|---|---|---|---|
| `AxiomContractError` | `500` | `CONTRACT_VIOLATION` | A bug, not user input. [Rules.md §3](Rules.md#rule-3--never-raise-on-bad-input-degrade): it "must terminate". Should not occur in a shipped build; if it does, the detail names the offending id, dimension or rank. CLI exit 1. |
| `IndexNotFoundError` — a requested `version` or `chunk_id` does not resolve | `404` | `VERSION_NOT_FOUND` / `CHUNK_NOT_FOUND` | Per-request scoping of an otherwise-healthy index. |
| `IndexNotFoundError` — nothing resolves and no `version` was named | `503` | `INDEX_UNAVAILABLE` | The server has nothing to serve. CLI exit 3, with `axiom index <repo> --version-id <id>` as printed remediation. |
| `DegradationExhaustedError` | `503` | `MODEL_UNAVAILABLE` | Every rung of a ladder failed. The one model condition [Setup.md §5.4](Setup.md#54-pre-download-for-an-offline-demo-do-this-the-day-before) wants to fail fast rather than hang on a socket timeout. CLI exit 1. |
| `ServeDependencyError` (`api/app.py`, subclasses `AxiomError`) | *(none — the server never starts)* | — | `fastapi`/`uvicorn` absent. Its `str()` is the install line and nothing else. CLI exit 1. |
| `AxiomError` — any other subclass | `500` | `INTERNAL_ERROR` | Backstop. CLI exit 1. |
| Pydantic `ValidationError` — request body or query/path parameter | `422` | `VALIDATION_ERROR` | Empty query, out-of-range `top_k`, malformed `version`/`chunk_id`. `detail` names the first failing field. |
| Pydantic `ValidationError` — `Settings` construction at startup | *(none — the process never binds)* | `CONFIG_ERROR` | Invalid configuration. CLI exit **2**; see §7.4. |
| Request body exceeds `AXIOM_API_MAX_BODY_BYTES` | `413` | `PAYLOAD_TOO_LARGE` | See §4. |
| Any other unhandled exception | `500` | `INTERNAL_ERROR` | Every stage is expected to either degrade (Rule 3) or raise one of the named classes. An `INTERNAL_ERROR` in the log is a defect to file, not a normal outcome. |

**Three conditions have no exception class, deliberately**, because raising for them would violate
Rule 3: a model that fails to load but has a rung left (degrades, logged at `WARNING`); a
tree-sitter parse failure (degrades down the chunker ladder); and an exhausted agent wall-clock
budget (returns best-so-far, surfaces as the `AGENT_BUDGET_EXCEEDED` warning in §5). Earlier drafts
of this section named `PrismModelError`, `PrismParseError` and `PrismBudgetError` for them; those
classes do not exist and should not be added — the ladder is the mechanism.

---

## 7. CLI Reference

The CLI is `typer`-based (`_CONTRACT.md §1`) and is the primary demo surface — the presenter's
worked commands in [Setup.md's verification ladder](Setup.md#8-verification-ladder) and
[TestPlan.md §7's manual test script](TestPlan.md#7-manual-test-script-demo-day) both run against
it directly, ahead of the API and the UI.

### 7.1 Entrypoint and subcommand names — settled

The entrypoint is **`axiom`** (`[project.scripts] axiom = "axiom.cli:app"`), per
[`ADR-015`](Decisions.md#adr-015--rename-prism-to-axiom) and
[`_CONTRACT.md §1`](_CONTRACT.md#1-runtime--toolchain-locked). There is no `prism` entrypoint. The
contract's §0/§1 have been corrected; the residue note this section used to carry is withdrawn.

Three name questions the suite used to leave open, each now closed against what `src/axiom/cli.py`
actually registers:

| Question | Answer | Why |
|---|---|---|
| `query` or `search`? | **`query`** | It is what the CLI registers. `axiom search` produces `No such command 'search'`. Every `axiom search` in any document is a defect, not an alias. |
| `versions` or `version`? | **`versions`** is canonical | `version` **is** registered, as a `hidden=True` alias, so `TC-069`'s `axiom version --help` passes and `TC-079`/`TC-080`'s `axiom version list|use|rm` work. Being hidden, it does not appear in `axiom --help` and must not be used in a demo or in documentation. |
| Is `gc` a tenth command? | Yes, and it is registered | `axiom gc` (blob garbage collection; `--dry-run` mutates nothing) is exercised by `TC-080` but is not among `FR-23`'s nine. Either `FR-23` grows to ten or `TC-080` loses its dependency — one of the two, and it is a `PRD.md` decision, not this document's. |

There is no `axiom config show`. [Setup.md §8](Setup.md#8-verification-ladder) rung 2 uses
`axiom --app-version` plus a one-line `get_settings()` dump instead.

### 7.2 Global flags

Registered on the top-level callback, so every subcommand accepts them. This is the full set, from
`axiom --help`:

| Flag | Type | Default | Effect |
|---|---|---|---|
| `--json` | flag | off | Machine-readable output: the same envelope models the HTTP API uses (§8). |
| `--profile NAME` | str | `AXIOM_PROFILE` (`default`) | Loads `configs/<profile>.yaml`. Valid: `default`, `demo`, `eval`, `fast`, `accurate`. |
| `--version ID` | **str** | active version | **The index version id to operate on — not the application version.** See `--app-version`. |
| `--app-version` | flag | — | Prints `axiom <version>` and exits 0. This is the one that answers "what am I running". |
| `--top-k N` | int | `AXIOM_TOP_K_DEFAULT` (`10`) | Final result count. |
| `--no-rerank` | flag | off | Disables the cross-encoder stage; results come back in RRF order with `score_field: "rrf_score"`. |
| `--no-agent` | flag | off | Exactly one retrieval pass, no refinement. |
| `--log-level LEVEL` | str | `AXIOM_LOG_LEVEL` (`INFO`) | `DEBUG`, `INFO`, `WARNING`, `ERROR`. |
| `--index-root PATH` | path | `AXIOM_INDEX_ROOT` (`.axiom`) | Root of the on-disk index tree ([`_CONTRACT.md §6`](_CONTRACT.md#6-on-disk-index-layout-locked)). |
| `--configs-dir PATH` | path | `configs/` | Directory holding the profile YAMLs. |
| `-h`, `--help` | flag | — | Usage. |

`--version` taking a string rather than being a version banner is the single most common surprise
on this surface, and it is deliberate: every subcommand needs to name an index version, and only
one needs to print a package version.

### 7.3 Subcommands

| Subcommand | Purpose | Key flags | Requirement |
|---|---|---|---|
| `axiom index <repo>` | Cold build of all three indexes over a repo tree | `--version-id ID`, `--force` (rebuild), `--parent ID`, `--no-activate`, `--json` | `FR-04`–`FR-10` |
| `axiom reindex` | Incremental reindex from a git diff | `--to REV`, `--from REV`, `--full` (force a full rebuild instead), `--version-id ID`, `--parent ID`, `--json` | `FR-18`, `FR-19` |
| `axiom query "<text>"` | Run the full retrieval pipeline and print ranked results | `--version ID`, `--all-versions`, `--top-k N`, `--type {semantic,structural,usage,hybrid}`, `--snippet-lines N`, `--json` | `FR-01`–`FR-14`, `FR-20`, `FR-21` |
| `axiom classify "<text>"` | Run only classification + planning; print the resulting `QueryPlan` | `--json` | `FR-01`, `FR-02`, `FR-03` |
| `axiom versions` | Registry operations: `list` (default), `use <id>`, `rm <id>` | `--purge` (with `rm`), `--json` | `FR-17`, `FR-20` |
| `axiom families` | List every `SnippetFamily` across indexed versions (browse, no query text) | `--version ID`, `--limit N`, `--multi-only`, `--diffs`, `--json` | `FR-21` |
| `axiom eval` | Run the MTEB adapter against `AppsRetrieval` | `--task NAME`, `--split {train,test}`, `--limit N`, `--json` | `FR-22`, `FR-26` |
| `axiom serve` | Start the FastAPI service (§2–§6) | `--host H`, `--port P` | `FR-24` |
| `axiom ui` | Start the Streamlit UI | `--port P`, `--api-base-url URL` | `FR-25` |

These nine are `FR-23`'s enumerated list. Two further commands are registered and are **not** among
them: `axiom gc` (blob garbage collection; `--dry-run` mutates nothing, otherwise deletes blobs no
surviving version references), exercised by [TestPlan.md `TC-080`](TestPlan.md); and the hidden
`axiom version` alias for `versions`. Both are documented here because real test cases depend on
them — see the table in §7.1.

Worked invocations, all verified against the registered CLI so this section cannot silently drift
from it:

```bash
# FR-23 canonical form
axiom query "How is the input preprocessed before the main function?" --top-k 5

# Setup.md verification ladder rung 5
axiom query "how is user input normalized before dispatch" --version smoke --top-k 3

# TestPlan.md TC-069 — CLI surface contract
axiom --help
axiom index --help
axiom query --help
axiom eval --help
axiom version --help        # the hidden alias; `axiom versions --help` is canonical

# TestPlan.md §6.1 — evaluation runs
axiom eval --task AppsRetrieval --limit 200      # smoke
axiom eval --task AppsRetrieval --split test     # reportable, full split

# Degraded-mode run (NFR-07)
AXIOM_LLM_ENABLED=false AXIOM_RERANKER_ENABLED=false \
  axiom query "normalize input" --version smoke --top-k 3
```

`axiom search` appears nowhere above, and should appear nowhere at all: it is not a registered
command.

### 7.4 Exit codes

Mirrors [Rules.md §9.2](Rules.md#92-error-taxonomy)'s error taxonomy exactly — the CLI and the API
share one exception hierarchy, so the mapping in §6 and the mapping below are two views of the same
table, not two independent ones.

| Exit code | Meaning | Source |
|---|---|---|
| `0` | Success | — |
| `1` | Internal error — `AxiomContractError`, `DegradationExhaustedError`, `ServeDependencyError`, any other `AxiomError`, or an unhandled exception reaching the CLI boundary | [Rules.md §9.2](Rules.md#92-error-taxonomy) |
| `2` | Usage error — an invalid or unrecognised flag ([TC-069](TestPlan.md)); a `Settings` `ValidationError`; a traversal-shaped `--version` value; or an empty/whitespace-only query ([TC-009](TestPlan.md)) | [Rules.md §9.2](Rules.md#92-error-taxonomy) |
| `3` | Index error — `IndexNotFoundError`, or an `OSError` reading the index tree. Printed with remediation: `axiom index <repo> --version-id <id>` | [Rules.md §9.2](Rules.md#92-error-taxonomy) |

**Exit 2 on an empty query is not a Rule 3 violation, and the two must not be conflated.** The CLI
gate short-circuits before the index is touched, exactly as the API's `min_length` gate returns
`422` before any file is opened — both are *boundary* checks on input that has not yet been
accepted. Rule 3 governs what happens after acceptance: a query that passes the gate and then goes
empty under normalisation degrades to an empty result set with `match_reason="empty_query"` and a
`stop_reason`, at exit 0. Two gates at two layers, deliberately; [Rules.md Rule 3](Rules.md#rule-3--never-raise-on-bad-input-degrade)
states the same split from the other side.

With `--json`, a failure prints `{"error": "<CODE>", "detail": "...", "remediation": "..."}` — the
same `error`/`detail` shape §6 defines for HTTP, plus an optional `remediation` line the HTTP
surface has no use for.

No usage message or remediation text printed on exit 2/3 ever includes a raw Python traceback,
matching the same no-stack-trace-in-the-response discipline stated for the HTTP API in §6.

---

## 8. CLI `--json` Output Shape

`--json` output reuses the same envelope models as the HTTP API — this is the literal meaning of
`FR-23`'s "every subcommand supports `--json` for machine-readable output" and `FR-24`'s "typed by
the same Pydantic models as the CLI."

| Subcommand | `--json` output |
|---|---|
| `axiom query` | Identical shape to `POST /v1/query`'s response body (§3.1) — all twelve fields, not the first six. `families` is always present and is empty unless `--all-versions` |
| `axiom classify` | The [`QueryPlan`](Schema.md#10-queryplan) object alone, unwrapped |
| `axiom versions` | Identical shape to `GET /v1/versions` (§3.2): `active_version`, `versions` |
| `axiom families` | `list[`[`SnippetFamily`](Schema.md#11-snippetfamily)`]` |
| `axiom index` | `IndexSummary`: `version_id`, `chunk_count`, `elapsed_ms`, `timings`, `file_count`, `embedding_model`, `embedding_dim`, `dense_backend`, `dense_index_kind`, `sparse_backend`, `structural_skipped`, `degradations` |
| `axiom reindex` | The incremental report — every `IndexSummary` field that still applies (`version_id`, `chunk_count`, `elapsed_ms`, `timings`, `degradations`) plus the carry-over accounting that is the whole point of the incremental path: `parent_version`, `strategy`, `files_chunked`, `files_dropped`, `chunks_carried`, `chunks_rebuilt`, `chunks_dropped`, `blobs_reused`, `blobs_written`, `embed_calls`, `missing_embeddings`. Not `IndexSummary`: forcing it into that shape would drop the reuse counters `FR-19` is measured by |
| `axiom eval` | The MTEB `TaskResult` shape written to `appsretrieval_results.json` ([`FR-22`](PRD.md#5-functional-requirements)), echoed to stdout |
| `axiom serve` / `axiom ui` | Not applicable — these are long-running processes; `--json` has no effect beyond structured startup logging ([Setup.md §7.1](Setup.md#71-core), `AXIOM_LOG_FORMAT=json`) |

In both index rows `timings` is a flat `dict[str, float]` keyed by the index-path stage tags in
[Rules.md §9.1](Rules.md#91-logging-discipline) (`walk`, `diff`, `chunk`, `dense`, `sparse`, `struct`,
`blob`, `write`, `index`, `manifest`); a stage that ran more than once is summed. The richer nested
per-stage record stays in the server log rather than on the wire.

Non-`--json` output for `axiom query`/`axiom classify`/`axiom versions`/`axiom families` is a
human-readable rendering of the identical underlying model — never a different set of fields, only
a different presentation. This is what makes a `--json` diff and a table read the same story, which
[NFR-08](PRD.md#6-non-functional-requirements) (determinism) and [NFR-10](PRD.md#6-non-functional-requirements)
(observability) both depend on.

---

## 9. Versioning and Stability of This Surface

Both the HTTP API and the CLI are **dev/demo surfaces, not a public API with a deprecation
policy** — this follows directly from [`NG-08`](NonGoals.md#ng-08--no-authentication-authorisation-or-multi-tenancy)
(no auth, no multi-tenancy) and [`NG-13`](NonGoals.md#ng-13--no-hosted-cloud-deployment) (nothing
hosted): there is no external, unknown consumer population to protect with a stability contract,
because the only consumers are the CLI itself, the Streamlit UI on `:8501`, and whoever clones the
repo and runs it locally. Inventing a formal `/v1/` → `/v2/` deprecation timeline, a
`Sunset`/`Deprecation` header policy, or a changelog-with-migration-guide process for this project
would be process theatre — there is no fleet of third-party integrations to migrate.

The `/v1/` path prefix is the one piece of forward-looking discipline this surface carries, and it
exists for exactly one reason, already stated in `NG-11`: **a future IDE-plugin client**, explicitly
out of scope for this hackathon cycle but explicitly named as the thing this frozen contract would
serve if built. `NG-11`'s own words: "make a future plugin a thin client... an editor integration
becomes a client of a frozen contract rather than a rewrite." The prefix is that seam, kept cheap
now (one path segment) in case it is ever needed later — it is not a promise that a `/v2/` is
planned, and no code in this project depends on the prefix meaning anything more than that.

If this contract does change before submission — a field renamed, an endpoint added, a status code
remapped — the change is recorded in [Decisions.md](Decisions.md) only if it reverses a locked
fact from `_CONTRACT.md` (none of the shapes in this document currently do); otherwise it is simply
edited here and the `Last updated` line above is bumped, consistent with every other doc in this
suite's convention of stating current facts rather than a version history of the contract.
