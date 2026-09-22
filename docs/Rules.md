# Rules

Binding engineering invariants for the PRISM codebase — violations block merge, not review comments.

**Owner:** Harshdeep Athawale
**Last updated:** 2026-09-15
**Status:** Draft

---

## How to read this document

Every statement here is imperative and binding. "Should" does not appear. If a rule is wrong,
change the rule in a PR that all four members approve — do not work around it in application code.
Rules are enforced in three places:

| Enforcement layer | What it catches | Where |
|---|---|---|
| Static (`ruff`, `mypy --strict`) | typing, imports, unused, formatting | CI step 1–2 |
| Test suite (`pytest`) | behavioural invariants, ID preservation, determinism | CI step 3 |
| Review | design, naming, config discipline, rule intent | human, see [§10](#10-code-review-rules) |

Terminology used below (`signal`, `stage`, `degradation ladder`, `candidate width`, `sufficiency
predicate`) is defined in [Glossary.md](Glossary.md#project-specific-jargon). The data model is
[Schema.md](Schema.md); the locked constants referenced here come from the technical contract and
are restated in [Design.md](Design.md) and [TechSpecifications.md](TechSpecifications.md).

---

## The Four Cardinal Rules

These four outrank every other rule in this document. If two rules conflict, the lower-numbered
cardinal rule wins.

### Rule 1 — IDs are sacred

> A `chunk_id` must survive preprocessing, chunking, indexing, retrieval, fusion, reranking and
> serialisation **byte-for-byte**. Never mutate, normalise, truncate, lowercase, strip, re-hash,
> prefix, suffix, or re-mint a `chunk_id` once it has been minted by the chunker.

`chunk_id` is minted exactly once, in `src/axiom/chunking/`, as the blake2b-128 hex digest of
`(normalised content ⊕ file_path ⊕ start_line)`. From that moment it is an opaque token. Every
downstream structure — `ScoredChunk.chunk_id`, `FusedResult.chunk_id`, `dense.idmap.json`,
`chunks.jsonl`, the `structural.sqlite` `symbols` table, the bm25s corpus ordering, the MTEB
adapter's output dictionary — carries the *same* string.

**Why this is cardinal, and why it is the most dangerous rule in the project:**

MTEB joins our retrieval output to the CoIR `AppsRetrieval` qrels by **exact string equality on the
document id**. When we evaluate on CoIR, the corpus `_id` *is* the id we must emit. There is no
fuzzy matching, no normalisation pass, no warning, and no exception. The failure mode is:

1. Some stage mangles the id — e.g. `chunk_id.strip()`, `str(int(chunk_id, 16))`, `chunk_id[:16]`,
   `f"{version_id}:{chunk_id}"`, or a `json.loads`/`json.dumps` round-trip through a key that got
   coerced to an int.
2. Retrieval still works. FAISS still returns neighbours. RRF still fuses. The reranker still
   scores. The API still returns plausible-looking results with real code in them.
3. MTEB looks up each returned id in `qrels`. Zero matches.
4. `NDCG@10` is computed as `0.0` for every query.
5. **No error is raised.** MTEB does not consider an unknown document id to be an error — an
   unknown id is simply a non-relevant document. A run that scores `0.0` is indistinguishable, from
   the inside, from a run where our retrieval is genuinely useless.
6. You have now burned two hours of evaluation wall-clock and will spend another two hours
   debugging retrieval quality when the bug is one `.strip()` call.

This has cost other teams entire submission windows. Treat any id transformation as a P0 defect.

**The defences, all mandatory:**

| Defence | Mechanism |
|---|---|
| Type-level | `chunk_id: str` everywhere. Never `int`, never `bytes`, never `UUID`. |
| Dict keys | Never use a `chunk_id` as a key in a structure that is serialised via a format that coerces keys. `chunks.jsonl` stores ids as values in a JSON object field, never as bare keys in YAML. |
| Identity test | `tests/test_id_integrity.py` (see [TestPlan.md](TestPlan.md), `TC-001`) asserts `set(ids_in) == set(ids_out)` across chunker → dense → sparse → structural → fusion → rerank for a fixture corpus. |
| Eval assertion | `src/axiom/eval/mteb_adapter.py` asserts every emitted id is a member of the corpus id set *before* handing results to MTEB, and raises `PrismContractError` if not. This is the one place where raising is correct — see Rule 3's exception list. |
| Version scoping | Cross-version scoping uses a **separate field** (`ChunkMetadata.version_id`), never a composite id string. |

**Corollary — id spaces:** PRISM has exactly two id spaces and they never mix.
`chunk_id` identifies a *chunk occurrence* (content + location). `content_hash` identifies *content
alone* and is the embedding-cache key and the cross-version dedupe key. `SnippetFamily.family_id` is
a third, derived id — it is minted in `src/axiom/versioning/evolutionary.py` and is never emitted to
MTEB.

### Rule 2 — Stages are pure

> For every stage `f`: same input + same config ⇒ same output. Byte-identical. Always.

A *stage* is any callable in the pipeline that takes typed input and produces typed output:
chunker, embedder, dense retriever, sparse retriever, structural retriever, fusion, reranker,
agent pass, evolutionary dedupe. Stages:

- take their configuration from an explicitly passed `Settings` object or a stage-local config
  model — never from module-level globals mutated at runtime, never from `os.environ` read inside
  the hot path;
- do not mutate their inputs (no in-place `list.sort()` on a caller-owned list, no assignment to a
  field of an input Pydantic model — Pydantic models in `src/axiom/schema/` are constructed
  `frozen=True` for exactly this reason);
- hold no cross-call mutable state that can change an output;
- do not read the wall clock, `random` without a seed, the filesystem outside the declared index
  root, or the network.

**The single sanctioned exception is the embedding cache** (`.axiom/blobs/<content_hash>.npy`). It is
sanctioned because it is *provably* result-neutral: the cache key is the blake2b-128 hash of the
normalised chunk content, and the cached value is the embedding of exactly that content produced by
the model named in `VersionManifest.embedding_model` with dim `VersionManifest.embedding_dim`. A
cache hit therefore returns the same vector the model would have produced. Two mandatory
consequences:

1. A cache entry is **invalidated by model identity, not by time**. If `embedding_model` or
   `embedding_dim` in the manifest differs from the current settings, the blob store for that
   version is not readable — `src/axiom/indexing/manifest.py` must refuse the load, not silently
   mix 384-dim and 1024-dim vectors.
2. Nothing else gets a cache. Not rerank scores, not query classifications during an eval run, not
   BM25 postings, not structural query results. Caching a rerank score keyed by query text would be
   result-neutral too, but it introduces an eviction policy, and an eviction policy is state that
   changes results. If a rerank cache is ever needed for demo latency it goes behind an explicit
   `AXIOM_RERANK_CACHE=true` flag that is **off** in `configs/eval.yaml`.

**Purity applies to the agent loop too.** The bounded refinement loop (max 2 passes) is pure with
respect to `(query, corpus, config, seed)`. It must not adapt across queries in an eval run — no
learned thresholds, no bandit over strategy weights, no memory of previous queries. An eval run
must produce the same JSON whether queries arrive in dataset order or shuffled.

### Rule 3 — Never raise on bad input; degrade

> A malformed, empty, adversarial, or pathologically long query must never terminate a run. The
> evaluation harness will hand us ~3,765 test queries. Dying at query 9,000 of a two-hour run is
> the single most expensive failure available to us.

Every stage declares a **degradation ladder**: an ordered list of fallbacks ending in a defined,
typed, empty-but-valid result. Degradation is logged at `WARNING` with the stage tag and the reason,
and recorded in the run's degradation counter so that the eval metrics log in
[Tracker.md](Tracker.md) reports how many queries degraded. Stage entrypoints below use the
canonical names from [Appflow.md](Appflow.md).

| Stage | Entrypoint | Bad input | Degradation ladder (top = preferred) |
|---|---|---|---|
| Query normalisation | `axiom.agent.loop:run` | empty / whitespace / control chars | strip control chars → if empty, return an empty `RetrievalResult` list with `match_reason="empty_query"` |
| Query classifier | `axiom.agent.classifier:classify` | LLM unavailable, LLM returns non-enum text | LLM classification → heuristic rule engine (identifier/keyword regexes) → `QueryType.HYBRID` with equal weights |
| Query planner | `axiom.agent.planner:build_plan` | LLM timeout, malformed JSON plan | LLM plan → single-sub-query plan containing the original query verbatim |
| Dense retrieval | `axiom.retrieval.dense:search` | query embedding fails, index missing | embed + search → empty `list[ScoredChunk]`, signal marked absent |
| Sparse retrieval | `axiom.retrieval.sparse:search` | tokeniser yields zero tokens (e.g. query is all punctuation) | code-aware tokenise → whitespace tokenise → empty list |
| Structural retrieval | `axiom.retrieval.structural:search` | no identifiers extractable, `structural.sqlite` missing | symbol/call lookup → identifier substring match → empty list |
| Chunker | `axiom.chunking.ast_chunker:chunk_file` | tree-sitter parse error, unsupported syntax | AST node boundaries → statement-boundary split → `axiom.chunking.fallback:split_text` fixed-window line split with overlap → whole file as one `ChunkKind.MODULE` chunk |
| Fusion | `axiom.retrieval.fusion:reciprocal_rank_fusion` | one or more signal lists empty | RRF over the non-empty lists, renormalising nothing (RRF needs no normalisation) → if all empty, empty result |
| Reranker | `axiom.rerank.cross_encoder:rerank` | model load failure, sequence too long | cross-encode → truncate to model max length and cross-encode → pass RRF order through unchanged (`rerank_score=None`) |
| Agent loop | `axiom.agent.loop:run` | wall-clock budget exhausted, a pass raises | return best results seen so far, `passes_used` recorded |
| Sufficiency check | `axiom.agent.evaluator:assess_sufficiency` | scores absent (rerank passthrough) | rerank-score predicate → RRF-score predicate → declare sufficient (never loop blindly) |
| Incremental reindex | `axiom.versioning.incremental:reindex` | `git diff` fails, no git repo | `axiom.versioning.gitdiff:diff_versions` → file-hash comparison against the parent `VersionManifest` → full reindex |
| Evolutionary dedupe | `axiom.versioning.evolutionary:build_families` | version metadata missing | family grouping → identity families (one member each) |

**Passthrough is a first-class outcome, not a bug.** When the reranker degrades to passthrough, the
result objects are still well-formed: `rerank_score is None`, `score` is the RRF score, and
`match_reason` records `"rerank_passthrough"`. A consumer can always tell degradation happened.

**Where raising *is* correct** — exactly three categories, all of which are programmer error or
contract violation, never input:

1. **Contract violations** — `PrismContractError`. An emitted id not in the corpus; an embedding of
   the wrong dimension; a manifest whose `embedding_model` disagrees with settings; a `rank` that is
   not 1-indexed contiguous. These mean the code is broken; failing loudly at run start is cheaper
   than a `0.0` score.
2. **Configuration errors** — `PrismConfigError`, raised at startup during `Settings` validation.
   Fail before the first query, never during query 9,000.
3. **Missing index artefacts at startup** — `PrismIndexError`. Raised by the index loader, once, at
   process start. Never raised per-query.

Everything else degrades. `except Exception` around a *whole stage* with a logged degradation and a
typed empty result is correct and encouraged. `except Exception: pass` is never correct (see
[§11](#11-anti-pattern-gallery), AP-04).

### Rule 4 — Higher is better; lists are sorted descending

> Every scorer in this codebase returns a value where larger means more relevant. Every ranked list
> is sorted descending by score. `rank` is 1-indexed and contiguous. No exceptions, no per-module
> polarity conventions, no "distance" variables.

FAISS `IndexFlatIP` / `IndexIVFPQ` on L2-normalised vectors returns inner product — already
higher-is-better, already equal to cosine similarity. BM25 is higher-is-better. The cross-encoder
emits a relevance logit or sigmoid — higher-is-better. Structural scores are graph-proximity scores
normalised to `(0, 1]` — higher-is-better. RRF is a sum of positive reciprocals — higher-is-better.

**Never introduce a distance.** If a library hands you an L2 distance, convert it at the library
boundary inside the adapter module, name the converted value `score`, and never let the distance
escape. A variable named `dist`, `cost`, `loss`, or `err` must not reach `ScoredChunk.score`.

**Tie-breaking is part of the sort contract**, because ties are common (identical BM25 scores,
identical structural hop counts) and unstable tie-breaks silently change NDCG@10 between runs. The
canonical sort key, implemented once in `src/axiom/core/` and used by every ranker, is:

```python
# src/axiom/core/ranking.py
def rank_key(item: ScoredChunk) -> tuple[float, str]:
    """Canonical descending sort key. Score descending, chunk_id ascending as a
    deterministic tie-break. Never change this without a recorded eval delta."""
    return (-item.score, item.chunk_id)

ordered = sorted(items, key=rank_key)
```

`chunk_id` is a hex digest, so the tie-break is arbitrary but *stable* and independent of insertion
order, dict iteration order, and thread scheduling. That is the whole point.

---

## 5. Determinism and seeding

1. `Settings.seed` (default `1337`) is the single seed. `src/axiom/core/` exposes `seed_everything()`
   which seeds `random`, `numpy.random`, and sets `PYTHONHASHSEED` guidance in docs (the process
   env must already be set; the CLI asserts it under `AXIOM_STRICT_DETERMINISM=true`).
2. ONNX Runtime session options set `intra_op_num_threads` and `inter_op_num_threads` from config.
   Thread count changes float reduction order; an unpinned thread count makes scores wobble in the
   4th decimal, which is enough to flip a tie and move NDCG@10 by ~0.1. Eval profiles pin threads.
3. FAISS `IndexIVFPQ` training is seeded. `nlist`, `m`, `nbits`, and `nprobe` live in config. A
   retrained index is a **new index kind instance** — record it in [Tracker.md](Tracker.md) with the
   eval delta before reporting any score built on it.
4. The LLM (`llama-cpp-python`) runs with `temperature=0.0`, fixed `seed`, and a fixed `n_ctx`.
   Non-zero temperature anywhere in the pipeline is prohibited.
5. Never iterate a `set` where the iteration order can reach an output. Sort first:
   `for cid in sorted(candidate_ids):`. `dict` preserves insertion order in 3.11, which is stable
   *given stable insertion*, which is exactly what set iteration does not give you.
6. `pytest` runs must pass with `-p no:randomly` and with random ordering. Order-dependent tests are
   defects in the test, not in the code.

## 6. CPU-only enforcement

1. No `torch.cuda` calls. No `.cuda()`, no `.to("cuda")`, no `device="cuda"`, no
   `torch.cuda.is_available()` branches — the branch itself is prohibited because it invites a code
   path that CI never exercises.
2. `CUDA_VISIBLE_DEVICES=""` is set in CI, in the Dockerfile, and in `scripts/run_eval.py`.
3. ONNX Runtime providers are explicitly `["CPUExecutionProvider"]`. Never rely on provider
   auto-selection.
4. A test (`TC-002`) greps the source tree for the forbidden substrings and fails the build on a
   hit. It is a crude check; it is also the check that will actually save us, because the failure it
   prevents ("works on my machine, dies in the judge's container") is unrecoverable on submission
   day.
5. `faiss-cpu` is the pinned dependency. `faiss-gpu` must never appear in `uv.lock`.
6. Any model added to the stack must have a measured CPU latency recorded in
   [Tracker.md](Tracker.md) before it is wired into a default profile.

## 7. Configuration discipline

> **Every tunable lives in `src/axiom/config.py` or a YAML profile under `configs/`. Nothing is
> hardcoded at a callsite. A reported score must be reproducible from a git SHA alone.**

That last clause is the entire rationale. On 24 September we will report an NDCG@10 number. Someone
— a judge, a reviewer, or us in a year — must be able to check out one commit, run one command, and
get that number. Every value that can move the number must therefore be in version control at that
SHA, and must be *findable*: one file, not scattered across 40 function defaults.

| Rule | Detail |
|---|---|
| Location | `Settings` (pydantic-settings v2) in `src/axiom/config.py`; profile YAMLs in `configs/` (`default`, `fast`, `accurate`, `eval`). |
| Precedence | CLI flag > env var (`PRISM_` prefix) > profile YAML > `Settings` field default. Documented once, in [Setup.md](Setup.md). |
| No magic numbers | `k=60`, `K_dense=100`, `K_sparse=100`, `K_struct=50`, `N_rrf=25`, `top_k=10`, `max_passes=2`, `wall_clock_budget_s=5.0`, `sufficiency_top1=0.35`, `sufficiency_floor=0.20`, `dedupe_cosine=0.95`, `stability_bonus=0.10` are **named config fields**. A literal `60` in `fusion.py` is a merge blocker. |
| Provenance | Every run writes the fully-resolved config (post-precedence) into the run artefact directory next to the results JSON. A results file without its resolved config is not a result. |
| No runtime mutation | `Settings` is frozen. Tests construct a new `Settings` rather than patching fields. |
| Docstring per field | Every `Settings` field carries a one-line description and its unit. Fields without units (`timeout`, `budget`, `size`) are ambiguous and are rejected in review; write `timeout_s`, `budget_ms`, `size_mb`. |
| New flag procedure | See [Contributing.md](Contributing.md#recipe-b--add-a-new-config-flag). |

## 8. The PLACEHOLDER convention

Many constants in the contract are *design estimates*, not measurements: the sufficiency thresholds
`0.35` / `0.20`, the dedupe cosine `0.95`, the stability bonus `0.10`, the `64–512` token chunk
target, the `16`-token merge floor. They were chosen from literature and intuition on day 1. They
are honest guesses and they must be *visibly* guesses until measured.

**The convention:**

```python
# src/axiom/config.py
sufficiency_top1_threshold: float = Field(
    default=0.35,
    description="Agent loop triggers refinement when top-1 rerank score is below this. "
                "# PLACEHOLDER — unvalidated, see T-141",
)
```

1. Any constant not yet validated against a held-out slice carries a literal `# PLACEHOLDER` marker
   in its declaration and an owning tracker task id.
2. `scripts/run_eval.py` scans the resolved config for `PLACEHOLDER` markers and prints a banner
   listing them at run start, and stamps `"placeholders": [...]` into the run artefact.
3. **A `# PLACEHOLDER` value may not appear in a reported score.** Before any number goes into the
   PPT, the release JSON, the eval metrics log, or a claim to the jury, every placeholder that can
   influence it must be replaced by a measured value, and the measurement must be logged in
   [Tracker.md](Tracker.md) as: constant name, old value, new value, dataset slice used, metric
   delta (`NDCG@10` before → after), and date.
4. Replacing a placeholder with the *same* number is a valid outcome — the point is the recorded
   delta, not the change. Mark it validated and delete the marker.
5. Removing a `# PLACEHOLDER` marker without a logged delta is a merge blocker.

## 9. Logging, errors, types, dependencies, and what may be committed

### 9.1 Logging discipline

1. `logging` via the project logger factory in `src/axiom/core/`. **No `print()` in `src/axiom/`.**
   `print` is allowed only in `scripts/` top-level output and `cli.py` user-facing output (which
   uses `typer.echo`/`rich`, not `print`).
2. Structured and stage-tagged. Every record carries `stage` (`chunk`, `embed`, `dense`, `sparse`,
   `struct`, `fuse`, `rerank`, `agent`, `version`, `eval`), and where applicable `query_id`,
   `version_id`, `elapsed_ms`, `n_candidates`.
3. Levels: `DEBUG` developer tracing; `INFO` stage boundaries and counts, one line per stage per
   query at most; `WARNING` every degradation, always with the ladder rung taken;
   `ERROR` contract/config/index failures only; no `CRITICAL`.
4. **No per-candidate logging at `INFO`.** 8,765 queries × 100 candidates × 3 signals is 2.6M lines
   and it will dominate eval wall-clock.
5. Never log full chunk text at any level above `DEBUG`. Log `chunk_id` and location.
6. Timing uses the `stage_timer` context manager so that the p50/p95 budgets in the contract are
   measured by the same clock everywhere.

### 9.2 Error taxonomy

All defined in `src/axiom/core/errors.py`, all deriving from `PrismError`.

| Exception | Raised when | Raised where | Caught where |
|---|---|---|---|
| `PrismError` | base, never raised directly | — | outermost CLI/API handler |
| `PrismConfigError` | invalid/contradictory settings | `config.py` validators, startup | CLI prints and exits 2 |
| `PrismContractError` | invariant violated (id not in corpus, wrong embedding dim, non-contiguous rank) | any stage, assertion sites | nowhere — it must terminate |
| `PrismIndexError` | index artefact missing/corrupt/version-mismatched | `indexing/manifest.py`, loaders | CLI prints remediation and exits 3 |
| `PrismModelError` | model load/inference failure | model adapters | the owning stage, which then degrades |
| `PrismParseError` | tree-sitter parse failure | `chunking/` | the chunker, which then degrades |
| `PrismBudgetError` | wall-clock budget exhausted | `agent/loop.py` | the loop, which returns best-so-far |

Rules: never raise bare `Exception`, `RuntimeError`, or `ValueError` from `src/axiom/`; never catch
`BaseException`; a caught exception is either re-raised, or logged with its type and a degradation
rung — never both swallowed and unlogged. Exception messages name the offending value and the
config field that controls it.

### 9.3 Type hints

1. Full annotations on every function and method in `src/axiom/`, including `-> None`.
2. `mypy --strict` is gating on `src/axiom/core`, `src/axiom/retrieval`, `src/axiom/schema`. These
   three are the shared blast radius; strictness there is non-negotiable. Other packages run
   non-strict mypy and must not *regress* the error count.
3. No `Any` in a public signature. Third-party untyped surfaces (`faiss`, `bm25s`, `tree_sitter`)
   are wrapped in a thin typed adapter module; `Any` stops at that boundary.
4. `# type: ignore` requires a specific error code and a trailing reason comment:
   `# type: ignore[no-untyped-call]  # faiss has no stubs`.
5. Pydantic models are the interchange format between stages. Do not pass `dict[str, Any]` between
   stages — if a stage needs a new field, add it to [Schema.md](Schema.md) and the model, with the
   four-member sign-off that schema changes require.
6. Python 3.11 syntax: `X | None`, `list[X]`, `StrEnum`. No `Optional`, no `typing.List`.

### 9.4 Dependency pinning

1. `uv.lock` is committed and authoritative. `requirements.txt` is generated from it and committed
   for the pip fallback path; regenerate both in the same commit or neither.
2. Direct dependencies are pinned to an exact version in `pyproject.toml`
   (`faiss-cpu==1.8.0`-style), not a range. Hackathon-scale project; a surprise minor bump on
   24 September is not a risk worth carrying.
3. Adding a dependency requires: a line in the PR body stating what it replaces or enables, a
   licence check (permissive only — MIT/BSD/Apache-2.0), an install-size note, and confirmation it
   has no GPU-only wheel on `linux/amd64`.
4. No dependency may be added after the day-9 feature freeze except to fix a submission-blocking
   defect.
5. Model weights are pulled at runtime into a gitignored cache directory, pinned by revision hash
   where the hub supports it. Never `main`/`latest`.

### 9.5 What may and may not be committed

| Never commit | Why |
|---|---|
| `data/` — corpora, CoIR downloads, sample repos | size; license; reproducible via `scripts/` |
| Model weights, ONNX exports, GGUF files | hundreds of MB; fetched by hash |
| `.axiom/` — any index artefact (`dense.faiss`, `sparse.bm25s/`, `structural.sqlite`, `blobs/`, `chunks.jsonl`) | derived; large; machine-specific |
| `*.npy`, `*.faiss`, `*.sqlite`, `*.db` anywhere | derived |
| `.venv/`, `__pycache__/`, `.mypy_cache/`, `.pytest_cache/`, `.ruff_cache/` | derived |
| `.env`, any token or API key | secrets |
| Notebooks with output cells | diff noise, embedded data |
| Large PNG/GIF screen recordings | put demo media in the release, not git history |

| Always commit | Why |
|---|---|
| `uv.lock`, `requirements.txt` | reproducibility |
| `configs/*.yaml` | a score must be reproducible from a SHA |
| Resolved-config artefacts for reported runs, and `appsretrieval_results.json` | evidence |
| Tiny fixtures under `tests/fixtures/` (< 64 KB each, hand-written JS) | deterministic tests |

`.gitignore` is the executable form of this table. If you find yourself typing `git add -f`, stop.

### 9.6 Performance regression policy

The budgets are contract, not aspiration: cold index 10k chunks ≤ 12 min; incremental reindex of
50 changed files ≤ 45 s; query p50 without agent loop ≤ 900 ms; query p95 with 2 agent passes ≤ 5 s;
peak RSS during query ≤ 4 GB — all on the 8-core / 16 GB reference box.

1. `scripts/bench_latency.py` prints p50/p95/peak-RSS for a fixed 50-query fixture set. It is the
   only number anyone quotes for latency.
2. Any PR that touches `retrieval/`, `rerank/`, `agent/`, or `indexing/` records before/after
   bench output in the PR body.
3. A regression of **> 10 %** on any budgeted metric, or any breach of an absolute budget, blocks
   merge. The fix is optimisation or an explicit, recorded trade against an accuracy gain — a
   latency regression bought with a measured NDCG@10 gain is a legitimate trade and belongs in
   [Decisions.md](Decisions.md) as an ADR.
4. Never buy accuracy with an unbounded loop. The agent's 2-pass / 5-second bound is a hard cap;
   widening it requires an ADR and a re-measured p95.
5. Never optimise on a guess. Profile (`cProfile`, `py-spy`) and put the measurement in the PR.

---

## 10. Code review rules

A reviewer is accountable for the rules, not for taste. Reviewing means running the checklist below,
not skimming the diff.

**A reviewer must check, in this order:**

1. **IDs** — does any line transform, reformat, slice, or re-key a `chunk_id`? Search the diff for
   `chunk_id` and read every occurrence.
2. **Purity** — does the change add module-level mutable state, a cache, a clock read, an
   unseeded random, or an in-place mutation of an input?
3. **Degradation** — does every new external call (model, file, parse, LLM) have a declared ladder
   rung and a `WARNING` log on the degraded path?
4. **Polarity** — does every new score go through `rank_key`? Is any new variable a distance?
5. **Config** — is every new number a `Settings` field? Is it documented with a unit? If
   unvalidated, is it marked `# PLACEHOLDER` with a tracker id?
6. **Types** — `mypy --strict` clean on the three strict packages; no new `Any` in a signature; no
   un-coded `type: ignore`.
7. **Schema** — does the diff touch `src/axiom/schema/`? If yes, is there four-member sign-off
   recorded in the PR?
8. **Tests** — is there a test that fails without the change? For a bug fix, does it reproduce the
   bug?
9. **Eval impact** — can this change move NDCG@10/MRR/Recall@100? If yes, is the delta recorded in
   [Tracker.md](Tracker.md)?
10. **Artefacts** — does the diff add data, weights, or `.axiom/` content?

**Blocks merge (no discussion, fix it):** a mutated `chunk_id`; a hardcoded tunable; `except: pass`;
`print` in `src/axiom/`; a CUDA reference; a failing test; a `mypy --strict` error in a strict
package; a committed data/weight/index artefact; an unremoved `# PLACEHOLDER` in a reported score; a
schema change without sign-off; a latency regression > 10 % without an ADR.

**Comment, does not block:** naming, docstring wording, test structure preference, micro-optimisation
suggestions without a profile, anything the reviewer would write differently but that violates no
rule.

**Etiquette:** review within 4 hours during the sprint. Say "blocking:" or "nit:" explicitly on
every comment. The author resolves, the reviewer closes. Self-merge is allowed only for
`docs/`-only changes.

---

## 11. Anti-pattern gallery

Twelve concrete pairs. Each one is a mistake that is cheap to make in this codebase and expensive to
find.

### AP-01 — Mutating a chunk id (Rule 1)

```python
# WRONG — strips, lowercases, or namespaces the id; MTEB now matches nothing and reports 0.0
results = {q_id: {c.chunk_id.strip().lower(): c.score for c in hits}}
results = {q_id: {f"{version_id}:{c.chunk_id}": c.score for c in hits}}
```

```python
# RIGHT — the id is opaque; version scoping lives in a separate field
results = {q_id: {c.chunk_id: c.score for c in hits}}
assert all(cid in corpus_ids for cid in results[q_id]), "id space violated"
```

### AP-02 — Fusing in score space instead of rank space (Rule 4, contract §5)

```python
# WRONG — mixes incomparable scales. BM25 is unbounded (0..30+), cosine is (-1..1).
# min-max normalising per query makes the result depend on the worst candidate in the list.
fused = 0.6 * norm(dense_scores) + 0.3 * norm(bm25_scores) + 0.1 * norm(struct_scores)
```

```python
# RIGHT — RRF consumes ranks only; no normalisation exists to get wrong.
def rrf(lists: dict[SignalKind, list[ScoredChunk]], weights: dict[SignalKind, float],
        k: int) -> list[FusedResult]:
    acc: dict[str, float] = {}
    contrib: dict[str, dict[SignalKind, int]] = {}
    for signal, ranked in lists.items():
        w = weights[signal]
        for item in ranked:                     # item.rank is 1-indexed
            acc[item.chunk_id] = acc.get(item.chunk_id, 0.0) + w / (k + item.rank)
            contrib.setdefault(item.chunk_id, {})[signal] = item.rank
    return [...]  # sorted by rank_key
```

### AP-03 — Unbounded agent loop (Rule 3, contract §5)

```python
# WRONG — "refine until good" has no upper bound on passes or on time.
# One pathological query stalls a 3,765-query eval run.
while not evaluator.is_sufficient(results):
    plan = planner.refine(plan, results)
    results = retrieve(plan)
```

```python
# RIGHT — bounded by passes AND wall clock; returns best-so-far on exhaustion.
deadline = monotonic() + settings.agent_wall_clock_budget_s
best = results
for pass_no in range(1, settings.agent_max_passes + 1):     # 2
    if evaluator.is_sufficient(best) or monotonic() >= deadline:
        break
    plan = planner.refine(plan, best)
    candidate = retrieve(plan)
    if top1(candidate) > top1(best):
        best = candidate
    log.warning("agent pass complete", extra={"stage": "agent", "pass": pass_no})
return best
```

### AP-04 — Silent exception swallowing (Rule 3)

```python
# WRONG — the run "succeeds" with a score of 0.0 and nothing in the logs explains why.
try:
    hits = self.structural_search(plan)
except Exception:
    pass
```

```python
# RIGHT — degrade to a declared rung, log it, count it.
try:
    hits = self.structural_search(plan)
except (PrismParseError, sqlite3.Error) as exc:
    log.warning("structural degraded to identifier match",
                extra={"stage": "struct", "reason": type(exc).__name__})
    self.degradations["struct.identifier_fallback"] += 1
    hits = self.identifier_match(plan)
```

### AP-05 — Non-deterministic iteration affecting tie-breaks (Rule 2, Rule 4)

```python
# WRONG — set iteration order varies with PYTHONHASHSEED and insertion history.
# Equal-scored candidates then land in different orders across runs and NDCG@10 wobbles.
for cid in candidate_ids:                 # candidate_ids is a set
    out.append(ScoredChunk(chunk_id=cid, score=scores[cid], ...))
out.sort(key=lambda c: -c.score)          # Python sort is stable, so input order leaks through
```

```python
# RIGHT — deterministic input order, and a total order in the sort key.
for cid in sorted(candidate_ids):
    out.append(ScoredChunk(chunk_id=cid, score=scores[cid], ...))
out.sort(key=rank_key)                    # (-score, chunk_id)
```

### AP-06 — Loading models at import time

```python
# WRONG — importing axiom.rerank now costs 900 ms and ~1.2 GB RSS.
# `axiom --help`, `pytest --collect-only`, and mypy all pay for it.
_SESSION = onnxruntime.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])

def rerank(query: str, chunks: list[Chunk]) -> list[FusedResult]: ...
```

```python
# RIGHT — lazy, explicit, injectable, and closable.
class CrossEncoderReranker:
    def __init__(self, settings: RerankSettings) -> None:
        self._settings = settings
        self._session: onnxruntime.InferenceSession | None = None

    def _ensure_session(self) -> onnxruntime.InferenceSession:
        if self._session is None:
            self._session = onnxruntime.InferenceSession(
                str(self._settings.model_path), providers=["CPUExecutionProvider"]
            )
        return self._session
```

### AP-07 — Hardcoded tunables at the callsite (Rule 7)

```python
# WRONG — the RRF constant, the widths, and the top-N are now invisible to config,
# unreproducible from a SHA, and impossible to sweep.
dense_hits = dense.search(qvec, 100)
sparse_hits = sparse.search(tokens, 100)
fused = rrf(lists, k=60)[:25]
```

```python
# RIGHT — every number is a named, documented, sweepable Settings field.
dense_hits = dense.search(qvec, cfg.candidate_width_dense)      # 100
sparse_hits = sparse.search(tokens, cfg.candidate_width_sparse)  # 100
fused = rrf(lists, weights=plan.strategy_weights, k=cfg.rrf_k)[: cfg.rrf_top_n]
```

### AP-08 — Mutating a caller-owned input (Rule 2)

```python
# WRONG — sorts the caller's list in place and edits an input model.
def rerank(self, query: str, candidates: list[FusedResult]) -> list[FusedResult]:
    for c in candidates:
        c.rerank_score = self._score(query, c)   # mutating a shared model
    candidates.sort(key=lambda c: -c.rerank_score)
    return candidates
```

```python
# RIGHT — build new frozen models; leave the input untouched.
def rerank(self, query: str, candidates: list[FusedResult]) -> list[FusedResult]:
    scored = [c.model_copy(update={"rerank_score": self._score(query, c)}) for c in candidates]
    return sorted(scored, key=lambda c: (-(c.rerank_score or 0.0), c.chunk_id))
```

### AP-09 — Cache keyed by anything other than content (Rule 2)

```python
# WRONG — keyed by path+mtime. A touched-but-unchanged file invalidates; a rename with
# identical content re-embeds; and the same key can map to two different vectors after a
# model swap, mixing 384-dim and 1024-dim arrays into one index.
key = f"{file_path}:{os.path.getmtime(file_path)}"
```

```python
# RIGHT — content hash, model-scoped, dimension-verified on load.
key = chunk.content_hash                       # blake2b-128 of normalised content
vec = blobs.load(key)                          # .axiom/blobs/<content_hash>.npy
if vec is not None and vec.shape[-1] != manifest.embedding_dim:
    raise PrismContractError(f"blob {key} dim {vec.shape[-1]} != {manifest.embedding_dim}")
```

### AP-10 — Turning a distance into a score by accident (Rule 4)

```python
# WRONG — FAISS L2 index returns squared distances: lower is better. Assigned straight to
# `score`, every ranking downstream is exactly inverted, and it still "works".
dists, idxs = index.search(qvec, k)
return [ScoredChunk(chunk_id=idmap[i], score=float(d), rank=r + 1, signal=SignalKind.DENSE)
        for r, (d, i) in enumerate(zip(dists[0], idxs[0]))]
```

```python
# RIGHT — L2-normalised vectors + IndexFlatIP/IndexIVFPQ give inner product == cosine,
# already higher-is-better. The polarity conversion, if any, never leaves this module.
sims, idxs = index.search(qvec, k)      # inner product, descending
return [ScoredChunk(chunk_id=idmap[i], score=float(s), rank=r + 1, signal=SignalKind.DENSE)
        for r, (s, i) in enumerate(zip(sims[0], idxs[0])) if i != -1]
```

### AP-11 — Raising on malformed query input (Rule 3)

```python
# WRONG — query 9,000 of a two-hour eval run is "   " and the run dies.
def tokenize(query: str) -> list[str]:
    tokens = CODE_TOKEN_RE.findall(query)
    if not tokens:
        raise ValueError(f"no tokens in query: {query!r}")
    return tokens
```

```python
# RIGHT — ladder down to whitespace splitting, then to empty, and log it.
def tokenize(self, query: str) -> list[str]:
    tokens = CODE_TOKEN_RE.findall(query)
    if tokens:
        return tokens
    tokens = query.split()
    if tokens:
        log.warning("sparse degraded to whitespace tokens", extra={"stage": "sparse"})
        return tokens
    log.warning("sparse yielded zero tokens", extra={"stage": "sparse"})
    return []
```

### AP-12 — Re-chunking or re-embedding on the read path (Rule 2, §9.6)

```python
# WRONG — the query path re-parses and re-embeds corpus files. p50 explodes from 900 ms to
# minutes, and the chunk ids produced here may not equal the indexed ones.
def search(self, query: str) -> list[RetrievalResult]:
    chunks = chunk_repo(self.repo_path)          # AST parse of the whole repo, per query
    vectors = self.embedder.encode([c.text for c in chunks])
    ...
```

```python
# RIGHT — the read path loads a built index; chunking and embedding are offline only.
def search(self, query: str) -> list[RetrievalResult]:
    qvec = self.embedder.encode_query(query)     # one short sequence
    hits = self.dense_index.search(qvec, self.cfg.candidate_width_dense)
    ...
```

### AP-13 — Filtering versions by string surgery instead of metadata

```python
# WRONG — parses semantics out of an opaque id, and breaks the moment version_id is a sha.
hits = [h for h in hits if h.chunk_id.startswith(version_id)]
```

```python
# RIGHT — filter on the declared metadata field.
hits = [h for h in hits if store.metadata(h.chunk_id).version_id == version_id]
```

### AP-14 — Reporting a number built on a placeholder (§8)

```python
# WRONG — the sufficiency thresholds were never measured; the number is not defensible.
# "NDCG@10 = 21.3" goes in the PPT with sufficiency_top1_threshold still marked PLACEHOLDER.
```

```markdown
RIGHT — Tracker.md entry before the number is quoted anywhere:

| T-141 | sufficiency_top1_threshold | 0.35 (PLACEHOLDER) → 0.42 (measured) |
| sweep on 300-query dev slice | NDCG@10 20.1 → 21.3 (+1.2) | 2026-09-21 | Harshdeep |
```

---

## 12. Rule change procedure

1. Open a PR that edits this file and nothing else.
2. State the rule id, the concrete incident or measurement that motivates the change, and the blast
   radius.
3. All four members approve. Cardinal rules 1–4 additionally require an ADR in
   [Decisions.md](Decisions.md).
4. Rules may not be relaxed during the final two days (day 9–10) of the sprint. If a rule is
   blocking submission on day 9, the answer is a recorded exception in the PR body naming the
   commit, not a rule edit.

## Related documents

| Document | Relationship |
|---|---|
| [Contributing.md](Contributing.md) | how to work the process these rules constrain |
| [Glossary.md](Glossary.md) | definitions of every term used above |
| [Schema.md](Schema.md) | the frozen data model the purity rules protect |
| [Design.md](Design.md) | stage boundaries the degradation ladder maps onto |
| [TestPlan.md](TestPlan.md) | `TC-###` cases that enforce these rules in CI |
| [Decisions.md](Decisions.md) | `ADR-###` records for every trade-off a rule allows |
| [Tracker.md](Tracker.md) | where placeholder replacements and eval deltas are logged |
| [Setup.md](Setup.md) | environment, config precedence, reference box |
