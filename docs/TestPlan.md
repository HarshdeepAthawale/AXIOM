# Test Plan

How PRISM is verified: what is unit tested, what is integration tested, what is only smoke tested, and what is measured by evaluation rather than asserted by a test.

**Owner:** Parth Deshmukh
**Last updated:** 2026-09-15
**Status:** Draft

Related: [_CONTRACT.md](_CONTRACT.md) · [Schema.md](Schema.md) · [Retrieval.md](Retrieval.md) · [Evaluation.md](Evaluation.md) · [Versioning.md](Versioning.md) · [Agent.md](Agent.md) · [Decisions.md](Decisions.md) · [Risks.md](Risks.md) · [OpenQuestions.md](OpenQuestions.md)

---

## 1. Testing Philosophy

PRISM is a ranking system. Ranking quality is a **measurement**, not an assertion. The single most
common way retrieval projects waste a 10-day sprint is writing brittle tests like
`assert results[0].chunk.metadata.symbol == "normalizeInput"`. That test fails the first time
someone retunes an RRF weight, and its failure carries no information about whether the system got
better or worse.

So we split responsibilities hard:

| Concern | Verified by | Never verified by |
|---|---|---|
| Schema shape, ID derivation, serialisation round-trips | `pytest` unit tests, strict | eyeballing JSON |
| Arithmetic (RRF, NDCG, MRR, stability bonus) | `pytest` unit tests against hand-computed constants | a golden file nobody re-derived |
| Invariants (sorted order, rank contiguity, dedup, determinism) | `pytest` property/invariant tests | manual demo |
| Wiring across module boundaries | integration tests on a fixture repo | unit tests with mocks of our own code |
| Whole-pipeline liveness, CLI, API, UI | smoke tests (does it run, is output well-formed) | assertions on which chunk ranked first |
| **Retrieval accuracy** | **MTEB run on CoIR `AppsRetrieval`, NDCG@10 / MRR / Recall@100** | **any `pytest` assertion** |
| Latency and memory budgets | `scripts/bench_latency.py` benchmark harness | a `time.time()` assert inside a unit test |

### 1.1 The pyramid

```
                      ┌──────────────────────────────┐
                      │  MTEB eval (nightly + gates) │  quality, not pass/fail
                      ├──────────────────────────────┤
                      │  Benchmarks (bench_latency)  │  budgets, threshold-gated
                      ├──────────────────────────────┤
                      │  Smoke: CLI / API / UI       │  ~10 tests, liveness only
                      ├──────────────────────────────┤
                      │  Integration: fixture repo   │  ~30 tests, real models off
                      ├──────────────────────────────┤
                      │  Unit: schema, hashing,      │  ~180 tests, fast, hermetic
                      │  chunking, fusion, metrics   │
                      └──────────────────────────────┘
```

### 1.2 Rules of engagement

1. **No unit test asserts a document's identity or position** unless the input is a synthetic
   fixture built specifically to make that ordering arithmetically forced (see TC-049..TC-052).
2. **No test downloads a model.** Real model weights appear only in `slow`-marked tests and in the
   benchmark/eval harnesses. Unit tests use `FakeEmbedder` (deterministic hash-seeded vectors,
   configurable dim) and `FakeCrossEncoder` (score = fixed lookup table).
3. **No test needs network.** `tests/conftest.py` sets `HF_HUB_OFFLINE=1`,
   `AXIOM_LLM_ENABLED=false`, and `TRANSFORMERS_OFFLINE=1` for the default run.
4. **Every test is full-suite-safe**: `tmp_path`-scoped `.axiom/` directories, no shared global
   state, no ordering dependency, no reliance on wall-clock except the three explicit budget tests.
5. **A test that pins a log string, a docstring, or a message template gets deleted.** We assert
   observable contract only: return values, file formats, exit codes, HTTP status and body shape.

### 1.3 Fixtures

| Fixture | Location | Contents |
|---|---|---|
| `fixture_repo` | `tests/fixtures/repo_v1/` | 14 JS files: 2 modules with exports, 1 class with methods, 1 file with a 5000-line generated function, 1 minified bundle, 1 syntactically broken file, 1 empty file, 1 file with only constants, circular import pair, duplicated file content at two paths |
| `fixture_repo_v2` | `tests/fixtures/repo_v2/` | `repo_v1` + 3 modified files, 1 added, 1 deleted, 1 pure rename, 1 rename-with-edit |
| `fixture_git_repo` | built by `tests/conftest.py` | real `git init`, two commits corresponding to v1 and v2, so `git diff --name-status` is exercised for real |
| `tiny_corpus` | `tests/fixtures/tiny_corpus.jsonl` | 40 hand-labelled (query, chunk_id) pairs for sanity-only relevance checks, never reported as a score |
| `rrf_vectors` | `tests/fixtures/rrf_cases.json` | ranked-list triples with hand-computed expected RRF scores |
| `FakeEmbedder` | `tests/fakes.py` | `blake2b(text)` → deterministic unit vector, dim configurable (384/1024) |
| `FakeCrossEncoder` | `tests/fakes.py` | scores from an explicit `dict[(query, chunk_id)] -> float`, default 0.5 |
| `FakeLLM` | `tests/fakes.py` | scripted responses queue; raises if called when `AXIOM_LLM_ENABLED=false` |

---

## 2. Test Categories

Categories mirror the package layout in [_CONTRACT.md](_CONTRACT.md) §3.

| Cat | Area | Module(s) under test | Owner |
|---|---|---|---|
| A | Query preprocessing: classification, identifier extraction, expansion, weighting | `agent/classifier.py`, `agent/planner.py` | Harshdeep Athawale |
| B | Chunking and snippet preprocessing | `chunking/` | Anish Grover |
| C | Indexing: dense/sparse/structural builders, manifest, blob store | `indexing/` | Prabinder Singh |
| D | Dense retrieval | `retrieval/dense.py` | Prabinder Singh |
| E | Sparse retrieval | `retrieval/sparse.py` | Prabinder Singh |
| F | Fusion (RRF) | `retrieval/fusion.py` | Prabinder Singh |
| G | Reranking | `rerank/cross_encoder.py` | Harshdeep Athawale |
| H | End-to-end pipeline, CLI, API, UI | `cli.py`, `api/`, `ui/` | Harshdeep Athawale |
| I | Versioning and incremental reindex | `versioning/gitdiff.py`, `versioning/incremental.py` | Parth Deshmukh |
| J | Evolutionary retrieval | `versioning/evolutionary.py` | Parth Deshmukh |
| K | Agent loop | `agent/loop.py`, `agent/evaluator.py`, `agent/llm.py` | Harshdeep Athawale |

Priorities: **P0** blocks the merge and blocks submission. **P1** blocks submission but not a
feature branch merge. **P2** is nice-to-have; may be deferred past 27 Sep.

---

## 3. Test Cases

### 3.1 Category A — Query preprocessing (TC-001 .. TC-011)

| ID | Cat | Description | Preconditions | Steps | Expected result | Pri | Auto |
|---|---|---|---|---|---|---|---|
| TC-001 | A | Semantic query classifies as `SEMANTIC` | `AXIOM_LLM_ENABLED=false`, heuristic classifier | `classify("How is the input preprocessed before the main function?")` | `QueryPlan.query_type == QueryType.SEMANTIC` | P0 | yes |
| TC-002 | A | Structural query classifies as `STRUCTURAL` | as above | `classify("Which files call tool XYZ before tool ABC?")` | `query_type == STRUCTURAL`; `extracted_identifiers` contains `XYZ`, `ABC` | P0 | yes |
| TC-003 | A | Usage query classifies as `USAGE` | as above | `classify("Where is the Bluetooth-settings deeplink used?")` | `query_type == USAGE` | P0 | yes |
| TC-004 | A | Multi-intent query classifies as `HYBRID` | as above | `classify("How does the auth module validate tokens before calling the API?")` | `query_type == HYBRID` | P1 | yes |
| TC-005 | A | Strategy weights match the locked table exactly | none | For each `QueryType`, build a plan; compare `strategy_weights` to the §5 table | Byte-exact match: SEMANTIC `{.6,.3,.1}`, STRUCTURAL `{.2,.2,.6}`, USAGE `{.25,.55,.2}`, HYBRID `{.34,.33,.33}` | P0 | yes |
| TC-006 | A | Weights always normalise to 1.0 | none | For all four types and for every override in `configs/*.yaml`, sum the three weights | `abs(sum - 1.0) <= 1e-9` | P0 | yes |
| TC-007 | A | Identifier extraction handles camelCase, snake_case, dotted, and quoted forms | none | `extract_identifiers("call handleDeeplink, parse_input, utils.normalize and 'MAX_RETRY'")` | Returns `["handleDeeplink","parse_input","utils.normalize","MAX_RETRY"]`, order preserved, no duplicates | P0 | yes |
| TC-008 | A | Identifier extraction emits no English stopwords | none | Run extractor over 50 natural-language queries in `tests/fixtures/queries.txt` | No returned identifier is in the stopword list; identifiers are never single-char unless quoted | P1 | yes |
| TC-009 | A | Empty and whitespace-only queries are rejected cleanly | none | `plan("")`, `plan("   \t\n ")` | Both raise `axiom.core.errors.EmptyQueryError`; no traceback escapes the CLI/API layer (CLI exit 2, API 422) | P0 | yes |
| TC-010 | A | 10,000-character query is truncated, not fatal | none | `plan("x" * 10000)` then run full retrieval | Plan built; query truncated to `settings.max_query_chars` (2048); pipeline returns a non-empty, well-formed result list; no exception | P1 | yes |
| TC-011 | A | Unicode and emoji in the query do not break tokenisation or embedding | `FakeEmbedder` | `plan("où est le déeplink 🔵 Bluetooth ?")` then retrieve | No `UnicodeError`; BM25 tokenizer yields ≥1 token; dense path produces a vector of the configured dim | P1 | yes |

### 3.2 Category B — Chunking and snippet preprocessing (TC-012 .. TC-024)

| ID | Cat | Description | Preconditions | Steps | Expected result | Pri | Auto |
|---|---|---|---|---|---|---|---|
| TC-012 | B | Function boundaries become chunk boundaries | tree-sitter installed | Chunk `tests/fixtures/repo_v1/src/utils/normalize.js` (3 top-level functions) | 3 chunks with `kind == FUNCTION`; each `location.start_line` equals the `function` keyword line; `end_line` equals the closing brace line | P0 | yes |
| TC-013 | B | Class methods get `kind == METHOD` and a `parent_symbol` | as above | Chunk `src/agents/bluetooth.js` (class `BluetoothAgent`, 4 methods) | 4 chunks `kind == METHOD`, `parent_symbol == "BluetoothAgent"`; 1 chunk `kind == CLASS` for the class shell | P0 | yes |
| TC-014 | B | `chunk_id` is stable across repeated chunking of identical input | none | Chunk the same file twice in separate processes; compare `chunk_id` sets | Sets identical; ids are 32-hex-char blake2b-128 | P0 | yes |
| TC-015 | B | **`chunk_id` round-trips unchanged through every stage** | fixture repo indexed | Capture the `chunk_id` set from the chunker; then read it from `chunks.jsonl`, from `dense.idmap.json`, from the bm25s doc-id map, from `structural.sqlite`, from each `ScoredChunk`, from each `FusedResult`, from each reranked result, and from the final `RetrievalResult.chunk.chunk_id` | All eight sets are equal for the retrieved subset; every retrieved id exists in `chunks.jsonl`; no stage rewrites, truncates, lowercases, or re-derives an id | P0 | yes |
| TC-016 | B | **`content_hash` is invariant under location change** | none | Hash a function's normalised text; move the same function to a different file and to a different line offset; hash again | `content_hash` identical in all three cases; `chunk_id` differs in all three (it binds `file_path` and `start_line`) | P0 | yes |
| TC-017 | B | **`content_hash` is sensitive to real content change** | none | Mutate the function body by one token (`>` → `>=`), one identifier rename, and one added statement | `content_hash` differs for each mutation; no collisions across the 3 mutants and the original | P0 | yes |
| TC-018 | B | `content_hash` normalisation ignores only declared-insignificant whitespace | none | Reformat a function: change indentation width, convert CRLF→LF, add a trailing newline | `content_hash` unchanged. Then delete a blank line *inside* a template literal → hash changes | P0 | yes |
| TC-019 | B | Oversized function splits at statement boundaries with 1-statement overlap | fixture with a 5000-line function | Chunk `src/generated/huge.js` | Multiple chunks, each ≤ 512 tokens except the last; every split point is a statement boundary (verify by re-parsing each chunk fragment); consecutive chunks share exactly one statement; union of chunk byte ranges covers the function with no gaps | P0 | yes |
| TC-020 | B | Chunks under 16 tokens merge into the parent | none | Chunk a file with a 2-line arrow function inside a class | No emitted chunk has `<16` tokens; the tiny function's text is present inside its parent chunk's `text` | P1 | yes |
| TC-021 | B | Syntactically broken file falls back to the regex identifier splitter | file with unbalanced braces | Chunk `src/broken/syntax_error.js` | No exception; ≥1 chunk emitted; `metadata.kind == BLOCK`; a `WARNING`-level log records the parse failure; `metadata.calls` populated from the regex fallback | P0 | yes |
| TC-022 | B | Empty file and a file with no functions produce zero or one MODULE chunk, never a crash | none | Chunk `src/empty.js` (0 bytes) and `src/constants.js` (only `const` exports) | Empty file → 0 chunks. Constants file → exactly 1 chunk with `kind == MODULE` | P0 | yes |
| TC-023 | B | Minified bundle is chunked without pathological blow-up | 1.2 MB single-line minified file | Chunk `src/vendor/bundle.min.js` with a 30 s timeout | Completes < 10 s; chunk count ≤ `settings.max_chunks_per_file` (2000); every chunk ≤ 512 tokens; no chunk has `end_byte < start_byte` | P1 | yes |
| TC-024 | B | Deeply nested closures do not exceed recursion limits | file with 200-deep nested arrow functions | Chunk `src/deep/nested.js` | No `RecursionError`; traversal is iterative; nesting beyond `settings.max_ast_depth` (64) is emitted as a single BLOCK chunk with a `WARNING` log | P1 | yes |

### 3.3 Category C — Indexing (TC-025 .. TC-034)

| ID | Cat | Description | Preconditions | Steps | Expected result | Pri | Auto |
|---|---|---|---|---|---|---|---|
| TC-025 | C | On-disk layout matches the contract exactly | `tmp_path` | `axiom index tests/fixtures/repo_v1 --version v1` | `.axiom/registry.json`, `.axiom/blobs/`, `.axiom/index/v1/{manifest.json,chunks.jsonl,dense.faiss,dense.idmap.json,sparse.bm25s/,structural.sqlite}` all exist; no extra top-level entries | P0 | yes |
| TC-026 | C | `dense.idmap.json` is a total bijection with the FAISS row space | index built | Load `dense.faiss` and the idmap | `index.ntotal == len(idmap)`; keys are `0..ntotal-1` with no gaps; values are unique; every value appears in `chunks.jsonl` | P0 | yes |
| TC-027 | C | Embeddings are L2-normalised | `FakeEmbedder`, dim 384 | Reconstruct 100 vectors from the FAISS index | `abs(norm - 1.0) <= 1e-5` for all | P0 | yes |
| TC-028 | C | Index kind is chosen by the 50k threshold | none | Build with 49,999 synthetic vectors, then 50,000 | `manifest.index_kind == "flat_ip"` then `"ivf_pq"`; the FAISS object type matches (`IndexFlatIP` / `IndexIVFPQ`) | P0 | yes |
| TC-029 | C | Manifest records model identity and dimension | `FakeEmbedder` dim 384 | Read `manifest.json` | `embedding_model` and `embedding_dim` match the active embedder; `chunk_count == len(chunks.jsonl)`; `created_at` is ISO-8601 | P0 | yes |
| TC-030 | C | Dimension mismatch between manifest and query embedder is a hard error | index built at dim 384 | Query with a dim-1024 embedder | `axiom.core.errors.IndexDimensionMismatch` raised before any FAISS call; message names both dims | P0 | yes |
| TC-031 | C | Blob store is content-addressed and deduped | two paths with byte-identical file content | Index the fixture repo | `.axiom/blobs/` contains exactly one `.npy` per distinct `content_hash`; the duplicated-content pair maps to one blob; embed call count == distinct content hashes | P0 | yes |
| TC-032 | C | Structural SQLite schema and referential integrity | index built | Query `structural.sqlite` | Tables `symbols`, `calls`, `imports`, `exports` exist; every `calls.caller_chunk_id` and `symbols.chunk_id` resolves in `chunks.jsonl`; `PRAGMA foreign_key_check` returns empty | P0 | yes |
| TC-033 | C | Circular imports are indexed without infinite traversal | fixture pair `a.js ↔ b.js` | Build the import graph | Completes; both edges present; graph builder visits each node once (assert visit counter) | P1 | yes |
| TC-034 | C | Index build is idempotent and atomic | index built once | Re-run `axiom index` on the same tree with the same version id; kill a build midway with SIGTERM and re-run | Re-run produces identical `manifest.json` except `created_at`; interrupted build leaves no partial version in `registry.json` (writes go to `index/<version>.tmp/` then `os.replace`) | P0 | yes |

### 3.4 Category D — Dense retrieval (TC-035 .. TC-041)

| ID | Cat | Description | Preconditions | Steps | Expected result | Pri | Auto |
|---|---|---|---|---|---|---|---|
| TC-035 | D | Returns exactly K candidates with 1-indexed contiguous ranks | index of 500 chunks | `dense_search(q, k=100)` | `len == 100`; `ranks == list(range(1,101))`; all `signal == DENSE` | P0 | yes |
| TC-036 | D | Scores are non-increasing | as above | inspect scores | `all(s[i] >= s[i+1])`; **descending-sort invariant** asserted by the shared `assert_ranked_list()` helper | P0 | yes |
| TC-037 | D | K larger than the corpus returns the whole corpus, not padding | index of 12 chunks | `dense_search(q, k=100)` | `len == 12`; ranks `1..12`; no `None` / sentinel chunk ids | P0 | yes |
| TC-038 | D | Exact-duplicate query of an indexed chunk's text retrieves that chunk at rank 1 | `FakeEmbedder` (hash-seeded, so identity is exact) | Embed chunk text verbatim as the query | Rank-1 `chunk_id` equals the source chunk; score `>= 0.999` | P0 | yes |
| TC-039 | D | Inner-product search is equivalent to cosine for normalised vectors | none | Compare FAISS scores to `numpy` cosine on the same 50 pairs | `max abs diff <= 1e-5` | P1 | yes |
| TC-040 | D | Version filter restricts results to one version | two versions indexed | `dense_search(q, version="v1")` | Every returned chunk's `metadata.version_id == "v1"` | P0 | yes |
| TC-041 | D | IVF-PQ `nprobe` is set from config and affects recall monotonically | 60k synthetic vectors | Search with `nprobe` 1, 8, 32; compare recall against the flat ground truth | Recall non-decreasing in `nprobe`; `nprobe` read from `settings.faiss_nprobe` (default 16) | P2 | yes |

### 3.5 Category E — Sparse retrieval (TC-042 .. TC-048)

| ID | Cat | Description | Preconditions | Steps | Expected result | Pri | Auto |
|---|---|---|---|---|---|---|---|
| TC-042 | E | Code tokenizer splits camelCase, snake_case, and dots while keeping the whole identifier | none | `tokenize("utils.handleDeeplink_v2")` | Contains `utils`, `handle`, `deeplink`, `v2`, **and** the intact `handledeeplink_v2`; lowercased | P0 | yes |
| TC-043 | E | Exact identifier query retrieves the defining chunk | fixture indexed | `sparse_search("handleDeeplink", k=100)` | The chunk defining `handleDeeplink` appears in the top 5 | P0 | yes |
| TC-044 | E | Ranked list obeys the descending-sort and contiguous-rank invariants | as above | `assert_ranked_list(results)` | Passes; `signal == SPARSE` on every element | P0 | yes |
| TC-045 | E | Out-of-vocabulary query returns an empty list, not an exception | as above | `sparse_search("zzqqxx_not_a_token")` | `[] `returned; downstream fusion handles the empty signal (see TC-055) | P0 | yes |
| TC-046 | E | Stopword-only query degrades gracefully | as above | `sparse_search("the of and is to")` | Returns `[]` or a list whose every score is `0.0`; pipeline overall still returns a non-empty result set via the dense signal | P1 | yes |
| TC-047 | E | **Regex fallback tokenizer is not vulnerable to ReDoS** | none | Feed the fallback tokenizer 12 adversarial inputs from `tests/fixtures/redos_inputs.txt` (e.g. `"a"*50000`, `"("*5000`, `("ab"*10000)+"!"`), each with a 2 s timeout | Every input completes < 200 ms; patterns are verified to contain no nested unbounded quantifier (static check over `sparse/tokenizer.py` pattern constants) | P0 | yes |
| TC-048 | E | bm25s index survives a save/load round-trip | index built | Save, load in a fresh process, re-run 20 queries | Identical `(chunk_id, score)` sequences to 1e-6 | P0 | yes |

### 3.6 Category F — Fusion (TC-049 .. TC-058)

This is the highest-value unit-test cluster in the project: it is pure arithmetic with a published
reference, so it can be pinned exactly.

| ID | Cat | Description | Preconditions | Steps | Expected result | Pri | Auto |
|---|---|---|---|---|---|---|---|
| TC-049 | F | **RRF arithmetic matches the hand-computed worked example** | weights `w_dense=w_sparse=w_struct=1.0` (unweighted mode, as in the Appendix B derivation) | Build three ranked lists: doc `A` at rank 5 in all three; doc `B` at rank 1 in two lists and absent from the third; doc `C` at rank 1 in one list and rank 10 in another. Fuse. | `score(A) == pytest.approx(3/65, abs=1e-6)` = 0.046154; `score(B) == approx(2/61)` = 0.032787; `score(C) == approx(1/61 + 1/70)` = 0.030679. Final order `[A, B, C]` — **A beats B**, which is the property that justifies RRF | P0 | yes |
| TC-050 | F | Weighted RRF equals the contract formula term by term | default SEMANTIC weights | Fuse the TC-049 lists with `{dense .6, sparse .3, struct .1}`; compute `Σ w_i/(60+rank_i)` by hand in the test | Every `rrf_score` matches the hand-computed value to 1e-9; `k` is read from config and defaults to 60 | P0 | yes |
| TC-051 | F | **Rank-space fusion is invariant to monotonic score rescaling** | none | Fuse three lists; then rescale each list's raw scores by a different strictly-increasing map (`s→10s+3`, `s→exp(s)`, `s→log1p(s+1)`), preserving order, and re-fuse | `rrf_score` for every doc is byte-identical between the two runs, and the output order is identical. Proves fusion never reads raw scores | P0 | yes |
| TC-052 | F | Fusion output order is fully determined and ties break deterministically | none | Fuse lists engineered so two docs tie exactly on `rrf_score` | Tie broken by ascending `chunk_id` (documented, not incidental); two runs give the same order; run in reversed input-list order → same output | P0 | yes |
| TC-053 | F | `contributions` records the true per-signal rank | none | Fuse the TC-049 lists | `contributions[DENSE] == 5` etc. for `A`; a signal in which the doc did not appear is **absent from the dict**, not `0` or `-1` | P0 | yes |
| TC-054 | F | `dominant_signal` is the signal with the largest weighted contribution | SEMANTIC weights | Doc at rank 1 in sparse, rank 40 in dense | `dominant_signal == SPARSE` because `.3/61 > .6/100`; test asserts the comparison, not a hard-coded signal | P0 | yes |
| TC-055 | F | **Empty-signal weight renormalisation** | none | Fuse with the structural list empty under SEMANTIC weights `{.6,.3,.1}` | Remaining weights renormalise to `{dense .6/.9 = .6667, sparse .3/.9 = .3333}`; the sum of effective weights is 1.0; scores equal a directly hand-computed two-signal fusion; a doc's score does not shrink merely because a signal was unavailable | P0 | yes |
| TC-056 | F | All signals empty yields an empty fused list, not a crash | none | Fuse three empty lists | Returns `[]`; caller (pipeline) converts this into an empty-but-valid `list[RetrievalResult]` and a `match_reason` of `"no candidates"` | P0 | yes |
| TC-057 | F | Fused list obeys the global ranked-list invariants and top-N truncation | 300 candidates across 3 lists | Fuse, then truncate | `assert_ranked_list` passes; output length `== min(25, unique_candidates)`; no duplicate `chunk_id` | P0 | yes |
| TC-058 | F | Candidate widths match the contract | fixture indexed | Run pipeline with tracing on | Dense requested K=100, sparse K=100, structural K=50, RRF→25, rerank→10; values come from config and are asserted against `_CONTRACT.md` §5 | P0 | yes |

### 3.7 Category G — Reranking (TC-059 .. TC-064)

| ID | Cat | Description | Preconditions | Steps | Expected result | Pri | Auto |
|---|---|---|---|---|---|---|---|
| TC-059 | G | Reranker reorders by `rerank_score` and preserves the candidate set | `FakeCrossEncoder` with a known score table | Rerank 25 fused candidates | Output is a permutation of the input's top-25 (no additions, no drops before truncation); sorted descending by `rerank_score`; `rrf_score` preserved on each item | P0 | yes |
| TC-060 | G | Reranker is applied to exactly top-N and returns top-10 | as above | Count cross-encoder calls | Exactly 25 pair scorings; final output length 10 | P0 | yes |
| TC-061 | G | Batching does not change results | as above | Rerank with `batch_size` 1, 8, 32 | Identical ordered `(chunk_id, rerank_score)` sequences | P0 | yes |
| TC-062 | G | Reranker failure degrades to the RRF order | `FakeCrossEncoder` raising `RuntimeError` | Run the pipeline | Non-empty results in pure RRF order; every `rerank_score is None`; `match_reason` notes the degradation; `ERROR` logged once, not per pair | P0 | yes |
| TC-063 | G | Long chunk text is truncated to the cross-encoder window, not rejected | chunk with 20k characters | Rerank | No exception; truncation at `settings.rerank_max_chars` (4096) applied to the document side only, never to the query | P1 | yes |
| TC-064 | G | Reranker fallback model swap keeps the interface | `AXIOM_RERANKER=cross-encoder/ms-marco-MiniLM-L-6-v2`, marked `slow` | Rerank 25 candidates | Same output type and invariants as primary; scores in a finite range; no dim/tokenizer assumptions leak | P1 | yes |

### 3.8 Category H — End-to-end pipeline, CLI, API, UI (TC-065 .. TC-072)

| ID | Cat | Description | Preconditions | Steps | Expected result | Pri | Auto |
|---|---|---|---|---|---|---|---|
| TC-065 | H | Full pipeline returns well-formed `RetrievalResult` objects | fixture indexed, fakes | `axiom query "how is input preprocessed" --json` | Valid JSON; each item validates against the `RetrievalResult` model; `chunk.location.start_line <= end_line`; `signals` non-empty; `score` finite | P0 | yes |
| TC-066 | H | **Determinism: same input + config + seed produces byte-identical output** | fixture indexed, `AXIOM_SEED=1337` | Run `axiom query ... --json > a.json` and again `> b.json`, in two separate processes | `a.json == b.json` byte for byte (after removing the single `elapsed_ms` field, which is excluded from the determinism contract and asserted separately to be present and positive) | P0 | yes |
| TC-067 | H | **Degradation with the LLM disabled produces a valid, non-empty result set** | `AXIOM_LLM_ENABLED=false`; `FakeLLM` asserts it is never called | Run the 12 demo queries | Every query returns ≥1 result; `FakeLLM.call_count == 0`; classification came from the heuristic engine; `QueryPlan.sub_queries == [original_query]` | P0 | yes |
| TC-068 | H | Degradation cascade: every declared fallback is individually exercised | none | For each of {embedder, reranker, LLM, tree-sitter} force the primary to raise, one at a time | Pipeline returns a non-empty well-formed result in all four cases; exactly one `WARNING`/`ERROR` per failure; a manifest/te­lemetry field records which fallback was active | P0 | yes |
| TC-069 | H | CLI surface contract | installed package | `axiom --help`, `axiom index --help`, `axiom query --help`, `axiom eval --help`, `axiom version --help` | Exit 0; the documented flags of [API.md](API.md)/[Setup.md](Setup.md) are all present; unknown flag → exit 2 with a usage message | P0 | yes |
| TC-070 | H | API contract: `POST /v1/query` | uvicorn on an ephemeral port, fixture index | Post `{"query":"...","top_k":10}` | 200; body has `results`, `query_plan`, `elapsed_ms`; `len(results) <= 10`; response validates against the `api/models.py` schema | P0 | yes |
| TC-071 | H | API input validation rejects malformed and hostile input | as above | Post empty query; `top_k=0`; `top_k=10000`; `version="../../etc"`; a 1 MB body | 422 for each (413 for the oversized body); no stack trace in the body; no filesystem access attempted (asserted with a patched `open`) | P0 | yes |
| TC-072 | H | Streamlit UI smoke: app imports and renders without a live index | none | `streamlit run src/axiom/ui/streamlit_app.py --server.headless true` for 20 s; also import-only test | Process stays alive; HTTP 200 on `/`; with no index present the UI shows an actionable "no index found" state rather than a traceback | P1 | partial |

### 3.9 Category I — Versioning and incremental reindex (TC-073 .. TC-080)

| ID | Cat | Description | Preconditions | Steps | Expected result | Pri | Auto |
|---|---|---|---|---|---|---|---|
| TC-073 | I | `git diff --name-status` parsing covers A/M/D/R | `fixture_git_repo` | `diff_versions(old_sha, new_sha)` | Returns the exact expected sets for added, modified, deleted; renames parsed as `R100 old new` with both paths captured; `R087`-style similarity scores handled | P0 | yes |
| triangle TC-074 | I | Only A and M files are re-embedded | fakes, embed-call counter | Incremental v1→v2 | Embed calls `== ` number of chunks in added+modified files whose `content_hash` is new; zero calls for D and for unchanged files | P0 | yes |
| TC-075 | I | **A pure rename costs zero embedding calls** | v2 differs from v1 only by `git mv` | Incremental reindex | `embed_call_count == 0`; all renamed chunks get new `chunk_id` (path changed) but reuse the existing blob by `content_hash`; `.axiom/blobs/` gains no new files | P0 | yes |
| TC-076 | I | **Incremental reindex is byte-equivalent to a full rebuild of the same tree** | none | (a) full-build v2 from scratch into `.prism_full/`; (b) build v1 then incremental to v2 into `.prism_inc/` | `chunks.jsonl` equal after sorting by `chunk_id`; idmap equal as a mapping; FAISS reconstructed vectors equal to 1e-6; bm25s query results identical over 20 probe queries; `structural.sqlite` equal after a canonical `SELECT ... ORDER BY` dump; `manifest.json` equal except `created_at` and `parent_version` | P0 | yes |
| TC-077 | I | A version with zero changes is a no-op | v2 == v1 | Incremental reindex v1→v2 | New manifest written with `parent_version == "v1"` and the same `chunk_count`; zero embed calls; zero new blobs; the index directory may be a hardlink/copy but must be independently readable | P1 | yes |
| TC-078 | I | A version that deletes every file yields a valid empty index | v2 is an empty tree | Incremental reindex | `chunk_count == 0`; `chunks.jsonl` is a 0-byte file (present, not missing); FAISS index has `ntotal == 0`; querying that version returns `[]` with no exception; blobs from v1 are retained (still referenced by v1) | P0 | yes |
| TC-079 | I | Registry tracks versions and the active pointer | two versions indexed | `axiom version list`, `axiom version use v1` | Both versions listed with chunk counts; active pointer switches; switching is O(1) (no reindex; assert no embed calls) | P0 | yes |
| TC-080 | I | Blob garbage collection removes only unreferenced blobs | 3 versions, then delete v2 | `axiom version rm v2` then `axiom gc` | Blobs referenced by v1 or v3 survive; blobs referenced only by v2 are deleted; `gc --dry-run` mutates nothing | P1 | yes |

### 3.10 Category J — Evolutionary retrieval (TC-081 .. TC-085)

| ID | Cat | Description | Preconditions | Steps | Expected result | Pri | Auto |
|---|---|---|---|---|---|---|---|
| TC-081 | J | **Family formation groups near-identical snippets** | 3 versions of `handleDeeplink` differing by one line each; `FakeEmbedder` configured to give pairwise cosine 0.97 | `build_families()` | Exactly one `SnippetFamily`; `len(members) == 3`; `versions` == the 3 ids; `members` sorted newest→oldest; `representative` is the newest member | P0 | yes |
| TC-082 | J | **Family formation does NOT group genuinely different snippets** | two functions with the same `symbol` in the same file path across versions but cosine 0.62; plus two near-identical functions with **different** `symbol` | `build_families()` | The 0.62 pair yields two families (below the 0.95 threshold). The same-content/different-symbol pair also yields two families — grouping requires matching `symbol` **and** `file_path`, not similarity alone | P0 | yes |
| TC-083 | J | Threshold boundary behaviour at cosine 0.95 | synthetic pairs at cosine 0.9499, 0.9500, 0.9501 | `build_families()` | 0.9500 and 0.9501 group (comparison is `>=`); 0.9499 does not. The comparison operator is asserted, not assumed | P0 | yes |
| TC-084 | J | `stability` is computed correctly and the 0.10 bonus applies only to multi-version families | 4 total versions; family present in 3 | Score with the evolutionary ranker | `stability == approx(0.75)`; `final == approx(base * 1.075)`; a single-version family (`stability == 0.25`) receives `final == base` (no bonus, because bonus requires `len(versions) >= 2`) | P0 | yes |
| TC-085 | J | Cross-version query returns one row per family, expandable to members | 3 versions indexed | `axiom query "deeplink" --all-versions` | No two results share a `family_id`; each result exposes its member list and per-member `diffs`; result count ≤ requested `top_k` | P1 | yes |

### 3.11 Category K — Agent loop (TC-086 .. TC-090)

| ID | Cat | Description | Preconditions | Steps | Expected result | Pri | Auto |
|---|---|---|---|---|---|---|---|
| TC-086 | K | **Loop terminates within `max_passes` and within the 5 s wall clock on adversarial input** | `FakeLLM` scripted to always answer "insufficient"; `FakeCrossEncoder` returning all scores 0.05 (never satisfies sufficiency); 10k-char query; empty-result corpus | Run `agent.loop.run()` 20 times | `passes_executed <= 2` every time; wall clock `< 5.0 s` every time; the loop exits with `stop_reason in {"max_passes","budget_exhausted","sufficient"}`; never raises, never loops forever | P0 | yes |
| TC-087 | K | Sufficiency thresholds fire exactly as specified | fake reranker with controlled scores | Case (a) top-1 = 0.34; (b) top-1 = 0.36 with 5 results > 0.20; (c) top-1 = 0.90 with only 2 results > 0.20 | (a) refine — top-1 < 0.35. (b) stop — sufficient. (c) refine — fewer than 3 results above 0.20. Thresholds read from config, defaults 0.35 / 0.20 | P0 | yes |
| TC-088 | K | Budget exhaustion mid-pass returns the best results so far, not an error | monkeypatched clock advancing 4.9 s during pass 1 | Run the loop | Returns the pass-1 result set; `stop_reason == "budget_exhausted"`; no partial/empty list; the check is before starting a new pass and before the rerank of a new pass | P0 | yes |
| TC-089 | K | Refinement changes the query and never repeats an identical retrieval | `FakeLLM` returning a rewrite | Run a 2-pass loop; capture the queries issued | Pass-2 query differs from pass-1; if the rewrite is identical to the original the loop stops early with `stop_reason == "no_new_query"` (guards against wasted passes) | P0 | yes |
| TC-090 | K | Results across passes are merged without duplicates and with stable provenance | 2-pass loop, overlapping candidate sets | Inspect the merged output | No duplicate `chunk_id`; each result records the pass that produced it; merge is order-stable; `assert_ranked_list` passes on the final list | P0 | yes |

**Count: 90 cases, TC-001 through TC-090, all unique.**

---

## 4. Edge-Case Catalogue

Each row is covered by the listed test case. No row is aspirational.

| Edge case | Where it bites | Covered by |
|---|---|---|
| Empty query | classifier, embedder (zero-length input) | TC-009 |
| Whitespace-only query | strips to empty after normalisation | TC-009 |
| 10,000-character query | embedder window, LLM prompt size, latency | TC-010, TC-086 |
| Query of only stopwords | BM25 returns nothing; fusion sees an empty signal | TC-046, TC-055 |
| Unicode / emoji in query | tokenizer, byte-offset math, terminal output | TC-011 |
| Minified JS file (one 1.2 MB line) | AST node count explosion, chunk count explosion | TC-023 |
| File with syntax errors | tree-sitter ERROR nodes | TC-021 |
| Empty file (0 bytes) | zero-chunk path, empty FAISS add | TC-022, TC-078 |
| File with no functions (constants only) | MODULE-kind fallback | TC-022 |
| Single 5000-line function | oversize split, overlap, token budget | TC-019 |
| Deeply nested closures (200 levels) | recursion limit, parent_symbol chains | TC-024 |
| Circular imports | import-graph traversal | TC-033 |
| Duplicate file content at two paths | content-hash dedup vs. chunk_id uniqueness | TC-016, TC-031 |
| Version with zero changes | incremental no-op path | TC-077 |
| Version that deletes every file | empty index, blob retention | TC-078 |
| Out-of-vocabulary identifier query | BM25 empty result | TC-045 |
| Adversarial regex input | ReDoS in the fallback tokenizer | TC-047 |
| Path-like `version` or `file_path` from the API | traversal (see [Security.md](Security.md)) | TC-071 |
| Cross-encoder raising mid-batch | rerank degradation | TC-062 |
| Interrupted index build | partial-state atomicity | TC-034 |

---

## 5. Performance Test Plan

### 5.1 Harness

`scripts/bench_latency.py` is the only sanctioned performance measurement. It is **not** a pytest
test; it emits a JSON row to `data/bench/<git_sha>.json` and a human table to stdout.

```bash
python scripts/bench_latency.py \
  --index .prism --version v1 \
  --queries tests/fixtures/bench_queries.txt \   # 100 queries, fixed order
  --repeat 5 --warmup 3 \
  --profile default \
  --out data/bench/
```

Rules:
- 3 discarded warm-up iterations per query (model load, mmap page-in, OS cache).
- 5 measured repeats; report **p50 and p95 across all (query, repeat) samples**, not per-query means.
- `psutil` samples RSS every 100 ms on the measurement thread; we report peak.
- Every row carries `git_sha`, `profile`, `model_id`, `chunk_count`, `index_kind`, `python`, `cpu`, `threads`.
- `OMP_NUM_THREADS` / `ONNXRUNTIME` intra-op threads pinned to 8 so results are comparable.

### 5.2 Reference hardware

The budgets in [_CONTRACT.md](_CONTRACT.md) §7 are defined on an **8-core x86-64 CPU with 16 GB
RAM, no GPU, SSD, Linux (Docker `python:3.11-slim-bookworm`)**. This is the box the numbers in the
PPT come from. Developer laptops (including arm64 machines) may be faster or slower; a laptop
measurement is never quoted as a budget result — it is used only for relative before/after
comparisons on the same machine. Every reported number states the box it came from.

### 5.3 Budgets and measurement

| ID | Budget (from §7) | Measured by | Method |
|---|---|---|---|
| PB-01 | Cold index, 10k chunks ≤ 12 min | `scripts/build_index.py --time-it` | Wall clock from process start to `registry.json` fsync, empty `.axiom/`, cold page cache (`vm.drop_caches` in the container-host script) |
| PB-02 | Incremental reindex, 50 changed files ≤ 45 s | `scripts/build_index.py --incremental --time-it` | Synthetic v2 with exactly 50 modified files (generator in `tests/fixtures/gen_v2.py`), warm model cache |
| PB-03 | Query p50 (no agent loop) ≤ 900 ms | `bench_latency.py --no-agent` | p50 over 100 queries × 5 repeats, model already loaded |
| PB-04 | Query p95 (2 agent passes) ≤ 5 s | `bench_latency.py --agent --force-passes 2` | p95 over the same set with the agent forced to the maximum pass count |
| PB-05 | Peak RSS during query ≤ 4 GB | `bench_latency.py --measure-rss` | Peak RSS of the worker process during PB-04 |

### 5.4 Regression gate

`scripts/bench_latency.py --compare data/bench/<baseline_sha>.json` exits non-zero when:

| Condition | Action |
|---|---|
| Any budget in §7 exceeded | **CI fails** (the `bench` job, nightly and on `main`) |
| p50 or p95 regressed > 15% vs. the last `main` baseline | **CI fails** |
| p50 or p95 regressed 5–15% | CI warns; a comment is posted on the PR; merge allowed |
| Peak RSS regressed > 10% | CI fails |
| Index build time regressed > 20% | CI warns |

The baseline is refreshed manually by the owner after an intentional, explained regression (e.g.
adding the structural signal). Refreshing requires a line in [Changelog.md](Changelog.md).

---

## 6. Evaluation Protocol

Retrieval quality is not a test. It is an experiment, and it follows experiment discipline.

### 6.1 Runs

| Mode | Command | Use |
|---|---|---|
| Smoke | `axiom eval --task AppsRetrieval --limit 200` | Did the harness wire up? Runs in ~3 min. |
| Dev | `axiom eval --task AppsRetrieval --limit 1000` | Directional signal while tuning. |
| **Reportable** | `axiom eval --task AppsRetrieval --split test` (full split, no `--limit`) | The only run whose number may be quoted anywhere |

### 6.2 Why a limited run is never a reportable score

`--limit N` truncates the **corpus and/or query set**. That changes the task, in two ways that both
inflate the number:

1. A smaller corpus means fewer distractors, so NDCG@10 rises mechanically — with 200 documents
   instead of 8,765, random ranking already scores far above its full-split expectation.
2. Truncation is not a random sample; it is the first N rows in file order, which correlates with
   dataset construction order.

So a `--limit` number is not a pessimistic estimate of the real score — it is a **different,
unrelated, usually higher** number. Rule: **any NDCG/MRR figure in the PPT, the README, the release
JSON, or a commit message must come from a full-split run, and must cite its git SHA.** A limited
run is labelled `SMOKE` in the experiment log and is never averaged with full runs.

### 6.3 Repeats and averaging

Our pipeline is deterministic given `AXIOM_SEED` (TC-066), so a repeat of the same SHA and config
should reproduce exactly. We therefore run:

- **1 run** to produce the number, plus **1 independent re-run** to confirm bit-identical metrics.
  If the two disagree, that is a determinism bug (fix it; do not average it away).
- **3 runs** only when a component is genuinely stochastic — currently just IVF-PQ training
  (k-means init). For `index_kind == "ivf_pq"` we report mean ± range over 3 index builds with
  seeds 1337/1338/1339.

### 6.4 Experiment log

`data/experiments.csv` is append-only, committed, one row per run:

| Column | Example |
|---|---|
| `run_id` | `2026-09-19T14:03Z-a1b2c3d` |
| `git_sha` | `a1b2c3d` (dirty tree → suffix `-dirty`, and the run is marked non-reportable) |
| `mode` | `FULL` / `SMOKE` |
| `task` / `split` | `AppsRetrieval` / `test` |
| `limit` | empty for FULL |
| `profile` | `eval.yaml` |
| `embedding_model`, `reranker`, `llm_enabled` | `Qwen3-Embedding-0.6B-int8`, `bge-reranker-v2-m3-int8`, `false` |
| `weights` | `dense=.6,sparse=.3,struct=.1` |
| `agent_passes` | `0` / `2` |
| `ndcg@10`, `mrr`, `recall@100`, `map` | `21.4`, `23.1`, `67.2`, `18.9` |
| `wall_clock_s`, `box` | `3120`, `ref-8c16g` |
| `notes` | "first run with structural signal" |

Discipline: **no number exists unless it has a row.** A screenshot is not a result. The submission
artifact `appsretrieval_results.json` must be reproducible from its row's SHA, profile, and command.

### 6.5 Quality gates by milestone

These gate the milestone, not the merge. See [Roadmap.md](Roadmap.md).

| Gate | Requirement |
|---|---|
| `0.2.0` dense baseline | Full-split NDCG@10 recorded, any value; harness proven |
| `0.3.0` hybrid + RRF | NDCG@10 > dense-only baseline by ≥ 2.0 absolute |
| `0.4.0` rerank | NDCG@10 ≥ 18.0 |
| `0.5.0` structural | No regression vs. `0.4.0`; Recall@100 ≥ 65.0 |
| `0.6.0` agent loop | NDCG@10 ≥ 20.0 (the §8 target) |
| `1.0.0` submission | NDCG@10 ≥ 20.0, MRR ≥ 22.0, all P0 tests green, all §7 budgets met |

---

## 7. Manual Test Script (Demo Day)

Run against the demo repo (`data/demo_repo/`, 3 versions), profile `default`, LLM enabled.
Executed by the presenter before recording and again before the live session. Each step has a
binary judgement — no "looks fine".

| # | Query / action | Expected qualitative behaviour | Pass criterion |
|---|---|---|---|
| M-01 | `axiom query "How is the input preprocessed before going to the main function?"` | Surfaces `normalizeInput` / `sanitizePayload` and the caller of `main()`. `match_reason` mentions the dense signal. | ≥2 of the top 5 are functions in the preprocessing chain, and the top result's file is one a human would open first. Fails if top 5 are unrelated UI code. |
| M-02 | `axiom query "Which files call tool XYZ before tool ABC?"` | Classified `STRUCTURAL`; results are files containing both calls; `signals` shows structural as dominant. | Every returned file genuinely contains both call sites, and ordering respects call order. Fails if any returned file lacks one of the two calls. |
| M-03 | `axiom query "Where is the Bluetooth-settings deeplink used?"` | Classified `USAGE`; BM25-dominant; returns the call sites, not the definition alone. | The definition and ≥1 call site both appear in the top 5. Fails if only the definition is returned. |
| M-04 | `axiom query "auth"` (one-word query) | Broad but non-empty; no crash; latency under 1 s. | ≥5 results, all auth-related. |
| M-05 | `axiom query ""` | Clean error message, exit code 2. | No traceback shown. |
| M-06 | `axiom query "$(python -c 'print("x"*10000)')"` | Truncation warning, still returns results within budget. | Returns results in < 5 s, no crash. |
| M-07 | `AXIOM_LLM_ENABLED=false prism query "How is the input preprocessed?"` | Identical shape of output, heuristic classification, visibly faster. | Non-empty results; output notes LLM-disabled mode. |
| M-08 | `axiom index data/demo_repo --version v3 --incremental --from v2` | Console shows changed-file count, re-embedded chunk count (much smaller than total), and elapsed time. | Elapsed < 45 s and re-embedded chunks < 20% of total. |
| M-09 | `axiom version list` then `axiom query "deeplink" --version v1` vs `--version v3` | Line numbers and code differ between versions in the way the git history says they should. | Both return results; the v1 and v3 snippets are visibly the historical and current forms. |
| M-10 | `axiom query "handleDeeplink" --all-versions` | One family row, expandable to 3 versions with per-version diffs, newest shown first. | Exactly one family for `handleDeeplink`; member order newest→oldest; diffs non-empty for ≥1 member. |
| M-11 | Streamlit UI: same query as M-01, click a result | Snippet with file path and line range, signal badges, expandable version history. | Clicking opens the snippet with correct line numbers matching the file on disk. |
| M-12 | Streamlit UI with the index deleted | Actionable empty state telling the user to run `axiom index`. | No traceback in the browser. |
| M-13 | Latency stopwatch on M-01 and M-02 | Sub-second for M-01, under 5 s for the agentic M-02. | Visible timer in the UI shows values within budget. |

Record the outcome of all 13 in the run-up checklist in [Tracker.md](Tracker.md) the day before the
demo. Any FAIL on M-01..M-03 or M-08..M-10 blocks recording the video.

---

## 8. CI Pipeline

`.github/workflows/ci.yml`, Ubuntu, Python 3.11 and 3.12 matrix.

| Stage | Command | Blocks merge | Typical time |
|---|---|---|---|
| 1. Lint | `ruff check . && ruff format --check .` | yes | 15 s |
| 2. Types | `mypy src/axiom/core src/axiom/retrieval src/axiom/schema` (strict) | yes | 40 s |
| 3. Fast tests | `pytest -m "not slow and not bench" --cov=src/axiom --cov-fail-under=<per §9>` | yes | ≤ 4 min |
| 4. Smoke index + query | `pytest -m smoke` — builds an index over `tests/fixtures/repo_v1` with fakes and runs 3 queries | yes | 60 s |
| 5. Dependency audit | `pip-audit -r requirements.txt --strict` | yes (see [Security.md](Security.md) §7) | 30 s |
| 6. Slow tests | `pytest -m slow` — real INT8 models, cached in the Actions cache | no (nightly + `main`) | ~15 min |
| 7. Benchmarks | `python scripts/bench_latency.py --compare <baseline>` | no on PRs; yes on `main` and nightly | ~10 min |
| 8. Eval smoke | `axiom eval --task AppsRetrieval --limit 200` | no (nightly); result appended to the experiment log | ~5 min |

Stages 1–5 are the merge gate and must stay under 7 minutes total. Anything slower gets a marker.

### 8.1 Marker convention

Declared in `pyproject.toml` under `[tool.pytest.ini_options] markers`:

| Marker | Meaning | Runs in the merge gate |
|---|---|---|
| (none) | Fast, hermetic, offline, < 1 s | yes |
| `smoke` | Exercises a real subsystem end to end with fakes; seconds | yes |
| `slow` | Downloads or loads real model weights, or > 10 s | no |
| `bench` | Timing/memory sensitive; must not run concurrently | no |
| `needs_git` | Shells out to a real `git` binary | yes (git is present on runners) |
| `xfail_known` | Tracks a known bug with a linked `OQ-##` or issue | yes, as xfail |

Enforcement: `pytest --strict-markers` and a `conftest.py` hook that fails any test taking > 1 s
without a `slow`/`bench`/`smoke` marker. That hook is what actually keeps the gate fast.

### 8.2 What a merge requires

1. Stages 1–5 green on both Python versions.
2. At least one P0 test covering the changed behaviour, or an explicit "no behaviour change" note in
   the PR description.
3. Schema changes (`src/axiom/schema/`): a [Changelog.md](Changelog.md) entry and a version bump
   per the breaking-change definition there.
4. Any change to fusion arithmetic, hashing, or the on-disk layout: the affected P0 test updated in
   the *same* commit, with its hand-computed expected value re-derived in the test body.

---

## 9. Coverage Targets

Line coverage is a smoke detector, not a goal. We set the bar high where a bug is silent and
catastrophic, and low where behaviour is measured by evaluation instead.

| Module | Line coverage target | Branch | Rationale |
|---|---|---|---|
| `src/axiom/schema/` | 95% | 90% | Pure data contract shared by four people. A wrong field name or a bad validator corrupts every artifact silently and is cheap to test exhaustively. |
| `src/axiom/retrieval/fusion.py` | 95% | 90% | Deterministic arithmetic with a published reference. There is no excuse for an untested branch; empty-signal renormalisation and tie-breaking are exactly where bugs hide. |
| `src/axiom/core/` (hashing, timing, errors, logging) | 90% | 85% | `chunk_id`/`content_hash` derivation is the backbone of dedup and incremental reindex. A hashing bug is invisible until the index is wrong. |
| `src/axiom/eval/metrics.py` | 90% | 85% | If NDCG is wrong, every number we report is wrong and we will not find out from the pipeline. |
| `src/axiom/versioning/` | 85% | 75% | Diff parsing and blob reuse are logic-heavy and testable; the equivalence test (TC-076) carries most of the weight. |
| `src/axiom/chunking/` | 80% | 70% | Heavy branching over AST shapes. Covering every grammar node is unbounded work; we cover the taxonomy (function/method/class/module/block), the oversize path, the fallback path, and the catalogued edge cases. |
| `src/axiom/agent/` | 75% | 65% | Termination, budget, and threshold logic are P0-tested (TC-086..TC-090). The *quality* of a rewrite is not testable — it shows up in NDCG. Chasing coverage here produces mock-echo tests. |
| `src/axiom/indexing/` | 75% | 65% | Correctness is asserted structurally (layout, bijection, dedup) rather than line by line; FAISS/bm25s internals are third-party and not ours to cover. |
| `src/axiom/rerank/` | 70% | 60% | The interesting behaviour is ordering and degradation (TC-059..TC-064); model internals are third-party. Rerank *quality* is an eval question. |
| `src/axiom/retrieval/{dense,sparse,structural}.py` | 70% | 60% | Thin adapters over FAISS/bm25s/SQLite. Invariant tests (rank contiguity, descending order, K-clamping) catch the real bugs; the rest is delegation. |
| `src/axiom/api/`, `src/axiom/ui/`, `src/axiom/cli.py` | 60% | — | Presentation layers. Contract-tested (TC-069..TC-072) and validated by the manual demo script; line coverage of Streamlit callbacks buys nothing. |
| **Project gate** | **80%** | — | `--cov-fail-under=80` in the merge gate |

### 9.1 Why retrieval-quality modules get a lower bar

Two different failure modes need two different tools.

A bug in `schema/` or `fusion.py` is a **logic defect**: it has a right answer, that answer is
computable by hand, and a passing test proves the code produces it. Coverage there is meaningful
because an uncovered branch is a branch whose right answer nobody checked.

A "bug" in `agent/` or `rerank/` is usually a **quality deficit**: the code ran correctly and
returned a worse ranking. No coverage percentage detects that, and the tests you write chasing 95%
in those modules are exactly the tests this plan bans — asserting that a mock was called, that a
prompt contains a substring, or that a particular chunk ranked first. Those tests break on every
retune and never catch a real regression.

So for those modules we test the **contract** (terminates, stays in budget, degrades, preserves
invariants, never crashes) at P0, accept 70–75% line coverage, and delegate the question "is it
good?" to NDCG@10 on the full split with a logged git SHA. That is the honest division of labour.
