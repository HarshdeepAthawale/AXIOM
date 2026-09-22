# API

HTTP endpoint contracts, error codes, CLI reference.

**Owner:** Harshdeep Athawale
**Last updated: 2026-09-16**
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

**Why the paths carry a `/v1/` prefix even though there is no versioning policy yet:** see §9. In
short, [`NG-11`](NonGoals.md#ng-11--no-ide-plugin) puts an IDE plugin out of scope for this cycle
but names this exact contract as the thing a future plugin would consume as "a thin client... of a
frozen contract rather than a rewrite." The prefix exists for that hypothetical future consumer;
it does not imply a `/v2/` is planned or that this doc promises compatibility across changes.

---

## 2. Base URL and Transport

| Setting | Env var | Default | Notes |
|---|---|---|---|
| Bind host | `AXIOM_API_HOST` | `127.0.0.1` | `0.0.0.0` inside Docker only, per [Deployment.md](Deployment.md) |
| Bind port | `AXIOM_API_PORT` | `8000` | [Setup.md §7.6](Setup.md#76-services) |
| Client base URL (used by the Streamlit UI) | `AXIOM_API_BASE_URL` | `http://127.0.0.1:8000` | [Setup.md §7.6](Setup.md#76-services) |

The full environment-variable reference is owned by [Setup.md §7](Setup.md#7-environment-variables);
this document names only the variables that directly shape the request/response contract.

Start the server with `axiom serve` (see §7). Every request and response body is `application/json`,
UTF-8, no trailing newline requirement. There is no CORS configuration — the surface is loopback-only
by default per §1, and adding permissive CORS to a same-origin-assumption dev server would widen the
trust boundary described in [Security.md](Security.md) for no benefit this project needs.

---

## 3. Endpoints

Four endpoints implement `FR-24`. All four are typed by Pydantic models: the payload models are the
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

| Field | Type | Description |
|---|---|---|
| `results` | `list[`[`RetrievalResult`](Schema.md#9-retrievalresult)`]` | `len(results) <= top_k` |
| `query_plan` | [`QueryPlan`](Schema.md#10-queryplan) | the plan from the final agent pass |
| `elapsed_ms` | `float` | wall-clock for the whole request; `> 0`; excluded from the determinism contract in [TestPlan.md TC-066](TestPlan.md) — every other field is byte-identical across repeat runs with the same seed, this one is not |
| `passes_used` | `int` | `1..AXIOM_AGENT_MAX_PASSES`; how many agent refinement passes actually ran (`FR-13`, `US-5`) |
| `timings` | `dict[str, float]` | per-stage elapsed ms, keyed by the stage tags in [Rules.md §9.1](Rules.md#91-logging-discipline) (`classify`, `embed`, `dense`, `sparse`, `struct`, `fuse`, `rerank`, `agent`, `format`); satisfies `NFR-10`'s auditability requirement |
| `warnings` | `list[object]` | degradation conditions that did not fail the request; see §5. Each item is `{"code": str, "detail": str}`. Empty list on a clean run |

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
  "timings": { "classify": 58.1, "embed": 34.7, "dense": 9.2, "sparse": 3.1, "struct": 12.4, "fuse": 1.6, "rerank": 601.0, "format": 8.9 },
  "warnings": []
}
```

**Status codes**

| Code | When |
|---|---|
| `200` | Always, for any well-formed, in-bounds request — including a query that degrades internally (empty result set, reranker passthrough, budget exhaustion). A degraded-but-valid answer is success, per [Rules.md Rule 3](Rules.md#rule-3--never-raise-on-bad-input-degrade). |
| `422` | Request fails validation: empty/whitespace `query`, `top_k` out of `[1, AXIOM_TOP_K_MAX]`, malformed `version` (including path-traversal-shaped values). See [TC-071](TestPlan.md). |
| `413` | Request body exceeds `AXIOM_API_MAX_BODY_BYTES` (default `262144`, 256 KiB — a real query is a few hundred bytes; the 1 MB adversarial body in [TC-071](TestPlan.md) is well past any legitimate use). This limit is API-specific and documented here rather than in [Setup.md](Setup.md)'s general environment-variable table because it governs request shape, not pipeline behaviour. |
| `503` | A required model is unavailable and cannot be fetched (`MODEL_UNAVAILABLE`, §6), or the index root itself is unreadable. |
| `500` | An internal contract violation (`PrismContractError`) reached the API boundary. Should never happen in a shipped build; see §6. |

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

### 3.3 `GET /v1/chunk/{chunk_id}`

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
| `404` | `chunk_id` is well-formed but not present in the resolved version's index (`CHUNK_NOT_FOUND`, §6) |
| `422` | `chunk_id` does not match the 32-hex-char pattern, or `version` is malformed |
| `503` | requested (or active) version's index is unreadable |

### 3.4 `GET /v1/health`

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
| `warmed` | `list[str]` | which components were actually constructed this call (`["embedder", "reranker"]`, plus `"llm"` when `AXIOM_LLM_ENABLED=true`); empty when `warm=false` or everything was already warm from a prior call |
| `elapsed_ms` | `float` | request wall-clock; on the first `?warm=true` call this is the 20-40 s cold-load cost the troubleshooting table describes, on every subsequent call (warm or not) it is near-zero because the singletons are already constructed |

Example — the canonical pre-demo warm call:

```bash
curl -s "http://127.0.0.1:8000/v1/health?warm=true"
```

```json
{ "status": "ok", "warm": true, "warmed": ["embedder", "reranker", "llm"], "elapsed_ms": 24531.8 }
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
`PrismContractError`/`PrismIndexError` territory.

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
| Reranker exceeds `AXIOM_RERANKER_TIMEOUT_MS` ([Setup.md §7.2](Setup.md#72-models)) | `200`, `warnings` entry `{"code": "RERANKER_TIMEOUT", "detail": "..."}`, RRF order returned unchanged, every `rerank_score: null` | [Rules.md](Rules.md)'s reranker degradation ladder: "cross-encode → truncate → pass RRF order through unchanged." A timeout is exactly this rung, not a failure. |
| Agent loop exhausts `AXIOM_AGENT_WALL_CLOCK_MS` ([Setup.md §7.3](Setup.md#73-agent-and-llm)) | `200`, `warnings` entry `{"code": "AGENT_BUDGET_EXCEEDED", "detail": "..."}`, best-results-so-far returned, `passes_used` reflects what actually ran | `PrismBudgetError` never escapes `agent/loop.py` in normal operation — [Rules.md §9.2](Rules.md#92-error-taxonomy) names it "raised where: `agent/loop.py`; caught where: the loop, which returns best-so-far." It is a warning by construction, not an exception that reaches the API layer. |
| Any first-stage signal (dense/sparse/structural) returns nothing | `200`, no dedicated warning entry (visible instead via `signals` on individual results and via `timings`) | Declared ladder rung, e.g. sparse "tokeniser yields zero tokens → empty list." An empty signal is routine, not exceptional — see the `TC-055` empty-signal renormalisation test. |
| A model load fails but a fallback is cached and loadable | `200`, no error; the fallback model's identity is visible in `GET /v1/health?warm=true`'s `warmed` list and in server logs at `WARNING` | `NFR-07` mandatory degradation: every model has a declared fallback. |
| A model is unavailable **and** no fallback is loadable (typically `AXIOM_OFFLINE=true` plus an uncached model) | `503`, `{"error": "MODEL_UNAVAILABLE", ...}` | There is no ladder rung left — see §6. This is the one model-loading condition Rule 3 does not ask to be swallowed, because [Setup.md §5.4](Setup.md#54-pre-download-for-an-offline-demo-do-this-the-day-before) deliberately wants it to fail fast and visibly rather than hang on a socket timeout in front of the jury. |
| `chunk_id` or `version` malformed, `top_k` out of range, empty `query`, oversized body | `422` / `413` | Input validation, not pipeline degradation — the request never reaches a stage. See §4. |
| An emitted id not in the corpus, a wrong embedding dimension, a non-contiguous rank | `500`, `{"error": "CONTRACT_VIOLATION", ...}` | [Rules.md §3](Rules.md#rule-3--never-raise-on-bad-input-degrade)'s one deliberately-fatal category: `PrismContractError` "means the code is broken; failing loudly... is cheaper than a 0.0 score." Should never occur in a shipped build. |

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
taxonomy (`src/axiom/core/errors.py`) to its HTTP treatment. Exception class names are shown as
`Prism*` because that is what [Rules.md §9.2](Rules.md#92-error-taxonomy) currently names them —
the [`ADR-015`](Decisions.md#adr-015--rename-prism-to-axiom) rename to `Axiom*` has not yet
propagated to that module; see the residue note at the end of §7.1.

| Exception | HTTP status | `error` code | Notes |
|---|---|---|---|
| `PrismConfigError` | *(none — process never binds)* | `CONFIG_ERROR` | Raised at `Settings` validation, startup only. The process exits non-zero before `uvicorn` starts listening, so no request ever observes this as an HTTP response; it is the CLI's exit code 2 case (§7.4). Listed here only because the CLI shares this error taxonomy. |
| `PrismContractError` | `500` | `CONTRACT_VIOLATION` | A bug, not user input. [Rules.md §3](Rules.md#rule-3--never-raise-on-bad-input-degrade): "nowhere — it must terminate." Should not occur in a shipped build; if it does, the detail names the offending id/dimension/rank. |
| `PrismIndexError` — index root missing/corrupt at startup | *(none — process never binds)* | `INDEX_UNAVAILABLE` | Raised by the index loader once, at process start (`indexing/manifest.py`). Mirrors CLI exit code 3. |
| `PrismIndexError` — a requested `version` or `chunk_id` does not resolve | `404` | `VERSION_NOT_FOUND` / `CHUNK_NOT_FOUND` | Per-request scoping of an otherwise-healthy index. This is the only `PrismIndexError` case that reaches a live request rather than blocking startup. |
| `PrismModelError` — degrades to a declared fallback | *(none — `200` with a warning or silently)* | — | Normal path per `NFR-07`; see §5. Not an HTTP error. |
| `PrismModelError` — no fallback loadable (typically `AXIOM_OFFLINE=true` + an uncached model) | `503` | `MODEL_UNAVAILABLE` | The one case [Setup.md §5.4](Setup.md#54-pre-download-for-an-offline-demo-do-this-the-day-before) wants to fail fast rather than hang. `detail` names the model repo id and notes `AXIOM_OFFLINE=true`. |
| `PrismParseError` | *(not applicable over HTTP)* | — | Chunker-only, index-time (`axiom index`/`axiom reindex`); there is no HTTP endpoint that triggers chunking. Degrades to the line-window fallback per [Rules.md §3](Rules.md#rule-3--never-raise-on-bad-input-degrade); see CLI exit codes (§7.4) for the index-build path. |
| `PrismBudgetError` | *(none — never escapes the loop)* | — | Surfaces only as the `AGENT_BUDGET_EXCEEDED` warning in §5. |
| Pydantic request validation failure | `422` | `VALIDATION_ERROR` | Empty query, out-of-range `top_k`, malformed `version`/`chunk_id` pattern. `detail` names the first failing field. |
| Request body exceeds `AXIOM_API_MAX_BODY_BYTES` | `413` | `PAYLOAD_TOO_LARGE` | See §4. |
| Any other unhandled exception | `500` | `INTERNAL_ERROR` | Backstop only; every stage is expected to either degrade (Rule 3) or raise one of the named `Prism*` classes. An `INTERNAL_ERROR` in the log is itself a defect to file, not a normal outcome. |

---

## 7. CLI Reference

The CLI is `typer`-based (`_CONTRACT.md §1`) and is the primary demo surface — the presenter's
worked commands in [Setup.md's verification ladder](Setup.md#8-verification-ladder) and
[TestPlan.md §7's manual test script](TestPlan.md#7-manual-test-script-demo-day) both run against
it directly, ahead of the API and the UI.

### 7.1 Entrypoint name, and the `query` vs `search` naming residue

The canonical entrypoint is **`axiom`**, per [`ADR-015`](Decisions.md#adr-015--rename-prism-to-axiom)
(PRISM → Axiom) and per every worked example in [Setup.md](Setup.md) and
[TestPlan.md](TestPlan.md), both of which already invoke it as `axiom`. `_CONTRACT.md §1` still
literally names the entrypoint `prism` — that is unpropagated rename residue, tracked as `T-201` in
[Tracker.md](Tracker.md), and this document follows the already-converged name rather than the
stale one, consistent with how [`OQ-04`](OpenQuestions.md#oq-04--project-name-collision-with-a-rival-submission)
itself handles the same class of residue.

A second, real naming inconsistency exists and is stated plainly rather than silently resolved:
**[PRD.md `FR-23`](PRD.md#5-functional-requirements) and [TestPlan.md `TC-069`](TestPlan.md) name
the query subcommand `query`.** [Schema.md §14.1](Schema.md#141-registryjson) independently
confirms `axiom versions` (not `axiom version`) as the listing command in its own prose. Those two
sources are this document's authority, so **`query` and `versions` are the canonical subcommand
names used throughout this section.** [Setup.md](Setup.md)'s worked examples (verification ladder
rung 5, and the `AXIOM_OFFLINE` troubleshooting-table example) invoke `axiom search` instead, and
predate `FR-23`'s naming; separately, [TestPlan.md `TC-079`](TestPlan.md) and
[`TC-080`](TestPlan.md) invoke `axiom version list` / `axiom version use` / `axiom version rm`
(singular) rather than `axiom versions ...`. This document does not silently pick a winner and
erase the discrepancy — both are real, observed inconsistencies in the current doc set, filed
alongside `T-201` in [Tracker.md](Tracker.md)'s `T-191`–`T-230` evaluation-discipline-and-submission
block, the same block that owns finishing the PRISM→Axiom propagation. Until that cleanup lands,
a reader following [Setup.md](Setup.md) verbatim will type `axiom search`/`axiom version` and get
the same behaviour documented here under `axiom query`/`axiom versions`.

### 7.2 Global flags

Every subcommand supports these, per `FR-23`:

| Flag | Type | Default | Effect |
|---|---|---|---|
| `--json` | flag | off | Machine-readable output: the same envelope models used by the HTTP API (§8). Non-`--json` output is a human-readable table/summary. |
| `--profile NAME` | str | `AXIOM_PROFILE` (`default`) | Loads `configs/<profile>.yaml`; overrides the environment for this invocation only. |
| `--index-root PATH` | path | `AXIOM_INDEX_ROOT` (`.axiom`) | Root of the on-disk index tree ([`_CONTRACT.md` §6](_CONTRACT.md)). |

### 7.3 Subcommands

| Subcommand | Purpose | Key flags | Requirement |
|---|---|---|---|
| `axiom index <repo>` | Cold build of all three indexes over a repo tree | `--version-id ID`, `--force` (rebuild), `--json` | `FR-04`–`FR-10` |
| `axiom reindex` | Incremental reindex from a git diff | `--to REV`, `--from REV`, `--full` (force a full rebuild instead), `--json` | `FR-18`, `FR-19` |
| `axiom query "<text>"` | Run the full retrieval pipeline and print ranked results | `--version ID`, `--all-versions`, `--top-k N`, `--type {semantic,structural,usage,hybrid}`, `--json` | `FR-01`–`FR-14`, `FR-20`, `FR-21` |
| `axiom classify "<text>"` | Run only classification + planning; print the resulting `QueryPlan` | `--json` | `FR-01`, `FR-02`, `FR-03` |
| `axiom versions` | Registry operations: `list` (default), `use <id>`, `rm <id>` | `--json` | `FR-17`, `FR-20` |
| `axiom families` | List every `SnippetFamily` across indexed versions (browse, no query text) | `--version ID`, `--json` | `FR-21` |
| `axiom eval` | Run the MTEB adapter against `AppsRetrieval` | `--task NAME`, `--split {train,test}`, `--limit N`, `--json` | `FR-22`, `FR-26` |
| `axiom serve` | Start the FastAPI service (§2–§6) | `--host H`, `--port P` | `FR-24` |
| `axiom ui` | Start the Streamlit UI | `--port P`, `--api-base-url URL` | `FR-25` |

These nine are exactly `FR-23`'s enumerated list. One additional command is exercised in
[TestPlan.md `TC-080`](TestPlan.md) but is **not** among `FR-23`'s nine — `axiom gc` (blob
garbage collection: `--dry-run` mutates nothing, otherwise deletes blobs unreferenced by any
surviving version). It is documented here for completeness since a real test case depends on it;
its absence from `FR-23` is the same class of residual documentation gap as §7.1's naming items,
and belongs in the same `T-191`–`T-230` cleanup pass.

Worked invocations, drawn directly from committed test contracts and worked examples so this
section cannot silently drift from them:

```bash
# FR-23 canonical form (this document)
axiom query "How is the input preprocessed before the main function?" --top-k 5

# Setup.md's worked form (pre-FR-23, see §7.1)
axiom search "how is user input normalized before dispatch" --version smoke --top-k 3

# TestPlan.md TC-069 — CLI surface contract
axiom --help
axiom index --help
axiom query --help
axiom eval --help
axiom version --help        # note: singular, see §7.1

# TestPlan.md §6.1 — evaluation runs
axiom eval --task AppsRetrieval --limit 200      # smoke, ~3 min
axiom eval --task AppsRetrieval --split test      # reportable, full split

# Degraded-mode run (NFR-07)
AXIOM_LLM_ENABLED=false AXIOM_RERANKER_ENABLED=false \
  axiom query "normalize input" --version smoke --top-k 3
```

### 7.4 Exit codes

Mirrors [Rules.md §9.2](Rules.md#92-error-taxonomy)'s error taxonomy exactly — the CLI and the API
share one exception hierarchy, so the mapping in §6 and the mapping below are two views of the same
table, not two independent ones.

| Exit code | Meaning | Source |
|---|---|---|
| `0` | Success | — |
| `1` | Unexpected/internal error — any exception not one of the named `Prism*` classes, or `PrismContractError` reaching the CLI boundary | typer's default for an uncaught exception; should not occur in a shipped build |
| `2` | Usage error — invalid/unrecognised flag ([TC-069](TestPlan.md)), or a `PrismConfigError` from `Settings` validation, or an empty/whitespace query ([TC-009](TestPlan.md)) | [Rules.md §9.2](Rules.md#92-error-taxonomy): "`PrismConfigError`... CLI prints and exits 2" |
| `3` | Index error — missing, corrupt, or version-mismatched index artefact | [Rules.md §9.2](Rules.md#92-error-taxonomy): "`PrismIndexError`... CLI prints remediation and exits 3" |

No usage message or remediation text printed on exit 2/3 ever includes a raw Python traceback,
matching the same no-stack-trace-in-the-response discipline stated for the HTTP API in §6.

---

## 8. CLI `--json` Output Shape

`--json` output reuses the same envelope models as the HTTP API — this is the literal meaning of
`FR-23`'s "every subcommand supports `--json` for machine-readable output" and `FR-24`'s "typed by
the same Pydantic models as the CLI."

| Subcommand | `--json` output |
|---|---|
| `axiom query` | Identical shape to `POST /v1/query`'s response body (§3.1): `results`, `query_plan`, `elapsed_ms`, `passes_used`, `timings`, `warnings` |
| `axiom classify` | The [`QueryPlan`](Schema.md#10-queryplan) object alone, unwrapped |
| `axiom versions` | Identical shape to `GET /v1/versions` (§3.2): `active_version`, `versions` |
| `axiom families` | `list[`[`SnippetFamily`](Schema.md#11-snippetfamily)`]` |
| `axiom index` / `axiom reindex` | An index-build summary object: `{"version_id": str, "chunk_count": int, "elapsed_ms": float, "timings": dict[str, float]}` — the same per-stage keys as §3.1's `timings`, plus indexing-only stages (`chunk`, `blob`) per [Rules.md §9.1](Rules.md#91-logging-discipline)'s stage tag list |
| `axiom eval` | The MTEB `TaskResult` shape written to `appsretrieval_results.json` ([`FR-22`](PRD.md#5-functional-requirements)), echoed to stdout |
| `axiom serve` / `axiom ui` | Not applicable — these are long-running processes; `--json` has no effect beyond structured startup logging ([Setup.md §7.1](Setup.md#71-core), `AXIOM_LOG_FORMAT=json`) |

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
