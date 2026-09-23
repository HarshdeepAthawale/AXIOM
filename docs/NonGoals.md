# Non-Goals

The explicit scope fence for Axiom: what we are deliberately not building in the re-baselined 2026-09-23 to 2026-09-27 window, why each exclusion is a decision rather than an oversight, and what we ship instead.

**Owner:** Parth Deshmukh
**Last updated:** 2026-09-23
**Status:** Draft

Related: [PRD.md](PRD.md) · [OpenQuestions.md](OpenQuestions.md) · [TechSpecifications.md](TechSpecifications.md) · [Design.md](Design.md) · [Decisions.md](Decisions.md) · [ImplementationPlan.md](ImplementationPlan.md) · [TestPlan.md](TestPlan.md) · [Security.md](Security.md) · [Deployment.md](Deployment.md) · [API.md](API.md) · [Glossary.md](Glossary.md)

---

## How to read this document

[PRD.md](PRD.md) is the positive statement of scope: `FR-01`–`FR-26` and `NFR-01`–`NFR-12` are
what Axiom does. This document is the negative statement. Everything listed here is a **Won't
have** for this cycle, per [PRD.md](PRD.md#7-prioritisation-moscow).

Three things this document is not:

- **Not a list of things we think are bad ideas.** Most entries are good engineering that simply
  does not earn a slot in a 10-day window scored on retrieval accuracy. Those live in
  [§4 Deferred, not rejected](#4-deferred-not-rejected).
- **Not a list of undecided questions.** If a thing is still being argued, it is an `OQ-##` in
  [OpenQuestions.md](OpenQuestions.md), not an `NG-##` here. An `NG-##` is settled.
- **Not negotiable mid-sprint.** Adding scope means deleting an `NG-##` in a PR that all four
  members approve, and recording an `ADR-###` in [Decisions.md](Decisions.md) that names what was
  dropped to pay for it. "While I was in there" is not an approval path.

Every entry carries exactly one **Category**, which is the single dominant reason for exclusion:

| Category | Meaning |
|---|---|
| out of problem scope | Theme 01 asks for ranked retrieval of existing code. This is a different product. |
| CPU budget | It does not fit the CPU-only constraint or the locked latency/throughput budgets. |
| time budget | It is feasible and useful, but it costs days we do not have before 2026-09-27. |
| risk | Building it makes the submission more likely to fail than not building it. |

---

## 1. The fence, in one sentence

> Axiom takes a natural-language query and a pre-indexed code corpus and returns a ranked list of
> **existing** code snippets, each located by file path and line range. It does not write code, does
> not explain code in prose, does not run code, and does not require a GPU, a network, an account,
> or a server it does not own.

---

## 2. Non-goal index

| ID | Non-goal | Category |
|---|---|---|
| [`NG-01`](#ng-01--no-code-generation) | No code generation | out of problem scope |
| [`NG-02`](#ng-02--no-answer-synthesis) | No answer synthesis | out of problem scope |
| [`NG-03`](#ng-03--no-natural-language-explanation-of-code) | No natural-language explanation of code | out of problem scope |
| [`NG-04`](#ng-04--no-language-support-beyond-javascript-demo-and-python-benchmark) | No language support beyond JavaScript (demo) and Python (benchmark) | time budget |
| [`NG-05`](#ng-05--no-gpu-cuda-rocm-or-metal-path) | No GPU, CUDA, ROCm, or Metal path | out of problem scope |
| [`NG-06`](#ng-06--no-model-fine-tuning-or-training-of-any-kind) | No model fine-tuning or training of any kind | CPU budget |
| [`NG-07`](#ng-07--no-distributed-or-multi-node-indexing) | No distributed or multi-node indexing | out of problem scope |
| [`NG-08`](#ng-08--no-authentication-authorisation-or-multi-tenancy) | No authentication, authorisation, or multi-tenancy | out of problem scope |
| [`NG-09`](#ng-09--no-persistent-user-accounts-history-or-personalisation) | No persistent user accounts, history, or personalisation | out of problem scope |
| [`NG-10`](#ng-10--no-production-sla-uptime-or-ha-guarantee) | No production SLA, uptime, or HA guarantee | out of problem scope |
| [`NG-11`](#ng-11--no-ide-plugin) | No IDE plugin | time budget |
| [`NG-12`](#ng-12--no-real-time-file-watching-or-indexing-daemon) | No real-time file watching or indexing daemon | time budget |
| [`NG-13`](#ng-13--no-hosted-cloud-deployment) | No hosted cloud deployment | risk |
| [`NG-14`](#ng-14--no-cross-repo-or-federated-search) | No cross-repo or federated search | out of problem scope |
| [`NG-15`](#ng-15--no-code-execution-sandboxing-or-dynamic-analysis) | No code execution, sandboxing, or dynamic analysis | risk |
| [`NG-16`](#ng-16--no-security-scanning-of-indexed-code) | No security scanning of indexed code | out of problem scope |
| [`NG-17`](#ng-17--no-proprietary-model-api-on-the-core-path) | No proprietary model API on the core path | risk |
| [`NG-18`](#ng-18--no-ann-index-on-the-benchmark-corpus) | No ANN index on the benchmark corpus | risk |
| [`NG-19`](#ng-19--no-hyde-in-the-default-path) | No HyDE in the default path | CPU budget |
| [`NG-20`](#ng-20--no-custom-react-or-spa-front-end) | No custom React or SPA front-end | time budget |
| [`NG-21`](#ng-21--no-python-313-support) | No Python 3.13 support | risk |
| [`NG-22`](#ng-22--no-generated-replacement-code-for-optimization-hints) | No generated replacement code for optimization hints | out of problem scope |
| [`NG-23`](#ng-23--no-llm-ingestion-of-retrieved-code) | No LLM ingestion of retrieved code | risk |
| [`NG-24`](#ng-24--no-non-git-version-sources) | No non-git version sources | time budget |
| [`NG-25`](#ng-25--no-learning-to-rank-or-click-feedback) | No learning-to-rank or click feedback | risk |
| [`NG-26`](#ng-26--no-indexing-of-build-output-vendored-or-non-source-assets) | No indexing of build output, vendored, or non-source assets | CPU budget |
| [`NG-27`](#ng-27--no-graph-database-for-the-structural-index) | No graph database for the structural index | time budget |
| [`NG-28`](#ng-28--no-native-windows-arm64-support) | No native Windows-arm64 support | risk |
| [`NG-29`](#ng-29--no-tuning-on-the-benchmark-test-split) | No tuning on the benchmark test split | risk |
| [`NG-30`](#ng-30--no-cross-query-result-cache) | No cross-query result cache | risk |

---

## 3. Non-goals in detail

### NG-01 — No code generation

**Category:** out of problem scope
**Boundary:** Axiom never emits a token of source code that was not read verbatim out of the
indexed corpus. No completion, no scaffolding, no test generation, no refactoring, no patch, no
diff-to-apply. Theme 01's task statement is "provide a ranking of code snippets in order of their
relevance to the query" — ranking, not authoring.
**Instead we do:** `FR-14` returns `RetrievalResult` records whose `chunk.text` is a byte-exact
slice of a real file, verifiable by `sed -n '<start_line>,<end_line>p' <file_path>`. `TC-###` cases
in [TestPlan.md](TestPlan.md) assert that round-trip equality, which is only possible because
nothing generative sits in the path.

### NG-02 — No answer synthesis

**Category:** out of problem scope
**Boundary:** Axiom is not RAG. It does not compose a paragraph that answers "how is the input
preprocessed?" by summarising retrieved chunks. The output is a ranked list, not a response. The
distinction matters because a synthesised answer is unverifiable against qrels: NDCG@10 is computed
over returned document ids, and an answer has no id.
**Instead we do:** return the 3-5 functions that constitute the answer, each with
`file_path:start_line-end_line`, and let the engineer read them. Persona P-2 in
[PRD.md](PRD.md#3-personas) fails us precisely if "the answer is a paragraph of prose instead of a
location".

### NG-03 — No natural-language explanation of code

**Category:** out of problem scope
**Boundary:** No "this function normalises the payload before dispatch" narration, generated or
templated beyond a fixed schema. The only prose Axiom emits per result is
`RetrievalResult.match_reason`, and that string explains **the retrieval decision**, not the code.
It is assembled from a fixed template over `FusedResult.dominant_signal`, the per-signal ranks in
`contributions`, and the matched `extracted_identifiers` — e.g. `structural: calls parseIntent
before dispatch (struct rank 2, dense rank 11)`.
**Instead we do:** make the ranking auditable rather than narrated. `FR-14` plus the per-signal
rank breakdown in the Streamlit result card (`FR-25`) shows *why this snippet*, which is the thing
a reviewer can check. See [Design.md](Design.md) for the formatter stage.

### NG-04 — No language support beyond JavaScript (demo) and Python (benchmark)

**Category:** time budget
**Boundary:** Exactly one tree-sitter grammar is shipped: `tree-sitter-javascript`. TypeScript,
JSX-beyond-what-the-JS-grammar-parses, Java, Kotlin, C++, and Go are all out. The two corpora are
not symmetric and this is deliberate, per [PRD.md](PRD.md#22-two-evaluation-contexts-two-profiles):
the **JavaScript** constraint in the problem statement describes the codebase being indexed for the
live demo; the CoIR `AppsRetrieval` benchmark corpus is **Python** solutions to English problem
statements. Python documents are indexed as opaque text by the dense and sparse signals only —
there is no Python grammar, no Python call graph, and none is needed, because APPS documents are
standalone single files with no cross-file structure to extract.
**Instead we do:** run two first-class profiles. `configs/eval.yaml` sets
`{dense 0.85, sparse 0.15, struct 0.0}` and disables the structural signal outright;
`configs/demo.yaml` carries the per-`QueryType` weight vectors and runs all three signals over
JavaScript. Any file the JS grammar cannot parse degrades to the line-window splitter (`FR-04`,
`NFR-07`) rather than aborting the index. Adding a grammar later is a config and chunker-registry
change, not an architecture change.

### NG-05 — No GPU, CUDA, ROCm, or Metal path

**Category:** out of problem scope
**Boundary:** Not "GPU optional" — GPU absent. No `torch.cuda` reference, no `device=` plumbing, no
accelerator extra in the dependency tree, no MPS on macOS even though the dev machines have it. A
CUDA-capable box must produce byte-identical results to the judge's laptop.
**Instead we do:** ONNX Runtime with the CPU execution provider exclusively, INT8 dynamic
quantisation for both the embedder and the cross-encoder, and a Q4_K_M GGUF query LLM through
`llama-cpp-python`. CPU torch is installed first and pinned (see [Setup.md](Setup.md) §3) so no
later resolution swaps in the CUDA build and its ~2.5 GB of unusable `nvidia-*` wheels. `NFR-06`
verifies this with `pip check` on a CPU-only image plus a smoke run in `python:3.11-slim-bookworm`.

### NG-06 — No model fine-tuning or training of any kind

**Category:** CPU budget
**Boundary:** Zero gradient steps. No contrastive fine-tune of the embedder on APPS train pairs, no
LoRA, no adapter, no distillation, no reranker fine-tune, no learned fusion weights via
backpropagation. On the reference box (8-core CPU, 16 GB RAM) a single epoch over 5,000 pairs with
a 0.6B encoder does not fit inside the build window, and a half-trained checkpoint is strictly
worse than the released one.
**Instead we do:** `FR-26` — supervised *configuration* tuning, not weight tuning. The 5,000
`AppsRetrieval` train pairs are split 4,000 tune / 1,000 dev (seeded, id lists committed under
`data/splits/`) and used to fit RRF signal weights per profile, the sufficiency thresholds
(0.35 / 0.20), and query-preprocessing variants.

**Retracted 2026-09-23: there is no accepted corpus-touching intervention.** The previous revision
named reverse doc2query document expansion in `axiom.indexing.expansion` as "the one accepted
accuracy intervention," banking an **estimated +4 to +8 NDCG@10**. That estimate was unsourced — it
appears in no paper cited anywhere in this suite and in no run of ours — and it was a quarter of the
stated path to the accuracy target. The module does not exist in `src/`, and the ADR that proposed
it is **withdrawn**: it required feeding chunk text to the query LLM, which
[ADR-008](Decisions.md#adr-008--the-query-llm-never-reads-code) and `_CONTRACT.md §2` forbid. See
[ADR-013](Decisions.md#adr-013--reverse-doc2query-expansion-as-the-one-corpus-touching-intervention)
for the full resolution. `NG-06` is therefore absolute for this cycle: **nothing writes to the
corpus.**

### NG-07 — No distributed or multi-node indexing

**Category:** out of problem scope
**Boundary:** No Ray, no Dask, no Celery, no shard-and-merge across machines, no work queue. One
process, one machine, one index root.
**Instead we do:** a bounded in-process worker pool for chunking and embedding on a single box.

**Correction, 2026-09-23.** The previous revision justified this with "cold index is **measured** at
636 s against the 720 s ceiling (`NFR-01`)." **No such measurement exists**, and the arithmetic in
[TechSpecifications.md §8.5](TechSpecifications.md#85-cold-index-placeholder-oq-14) projects
20–40 min, not 636 s: 10k chunks x ~256 tokens through Qwen3-Embedding-0.6B needs ~3.24 TOPS
sustained against a ~5.1 TOPS *theoretical* CPU peak. The 636 s figure is withdrawn; `NFR-01` is
`# PLACEHOLDER`, tracked as `OQ-14`.

The non-goal survives the correction, and it is worth saying why rather than quietly keeping it.
Distribution is excluded on **scope and time**, not on headroom: Ray/Dask/Celery is a day of work
plus a failure mode the jury's laptop cannot reproduce, and it would not be reached for even if the
index took an hour. The honest levers if `OQ-14` confirms the projection are in §8.5 of
TechSpecifications — a shorter max-token bound, or `all-MiniLM-L6-v2` for the demo index — none of
which is a cluster.

### NG-08 — No authentication, authorisation, or multi-tenancy

**Category:** out of problem scope
**Boundary:** No login, no API key, no JWT, no RBAC, no per-tenant index isolation, no rate limiter.
`FR-24`'s FastAPI service is explicitly dev-only and unauthenticated. This is not an oversight to be
fixed with middleware; it is a deployment posture.
**Instead we do:** bind the API to loopback, document it as a development and demo surface only, and
state the trust boundary plainly in [Security.md](Security.md). Anyone who can reach port 8000 can
read the index, and anyone who can read the index can already read the repo it was built from — so
the API grants no privilege that the filesystem did not already grant.

### NG-09 — No persistent user accounts, history, or personalisation

**Category:** out of problem scope
**Boundary:** No user records, no saved searches, no "recent queries", no per-user relevance
profile, no telemetry upload. A query carries no identity and leaves no trace beyond a log line.
**Instead we do:** keep every query stateless, which is what makes `NFR-08` (determinism) testable:
identical query + identical index + identical config yields byte-identical `chunk_id` ordering,
because there is no per-user state to perturb it. The Streamlit session holds the current result set
in memory and nothing else.

### NG-10 — No production SLA, uptime, or HA guarantee

**Category:** out of problem scope
**Boundary:** We publish no availability target, no error-rate budget, no graceful-restart story, no
health-check-driven orchestration beyond a `GET` health endpoint. Axiom is a hackathon prototype
that must survive a 10-minute jury session, not a service with an on-call rotation.
**Instead we do:** state latency as an *engineering budget*, not a promise — and state whether the
budget has been measured.

**Correction, 2026-09-23.** The previous revision called 768 ms "the canonical query p50," broken
down as classify 60 + query-embed 35 + three signals concurrent 28 + RRF 3 + hydrate 12 + rerank
25 pairs 620 + format 10, "leaving 132 ms of headroom." **No run produced any of those numbers**,
and the 620 ms rerank line was `ms-marco-MiniLM-L-6-v2`'s figure written against
`bge-reranker-v2-m3`'s row — off by roughly 25x, with the arithmetic shown in
[TechSpecifications.md §8.1](TechSpecifications.md#81-why-the-reranker-line-was-wrong-the-arithmetic).
The 768 ms p50 and the 132 ms headroom are **retracted**. This document no longer restates a latency
breakdown at all: [TechSpecifications.md §8](TechSpecifications.md#8-latency-and-memory-budget-placeholder-unmeasured)
is the single home for it, every row is `# PLACEHOLDER` with an owning `OQ-##`, and per
[Rules.md §8](Rules.md#8-the-placeholder-convention) none may be quoted until its measurement lands.

What survives unchanged is the *discipline*: `NFR-10` requires every stage to emit a timing record
and every `--json` response to carry a `timings` block, so the moment a number exists it is auditable
from a single run rather than asserted in prose. That was always the load-bearing claim; the 768 ms
was not.

### NG-11 — No IDE plugin

**Category:** time budget
**Boundary:** No VS Code extension, no JetBrains plugin, no LSP server, no editor-embedded result
panel. Packaging, marketplace metadata, and an extension host debug loop are two days that buy zero
rubric points against `FR-25`'s Streamlit UI.
**Instead we do:** make a future plugin a thin client. `FR-23` gives every CLI subcommand a `--json`
mode with stable field names and non-zero exit on failure; `FR-24` exposes the same Pydantic models
over HTTP, documented in [API.md](API.md). An editor integration becomes a client of a frozen
contract rather than a rewrite.

### NG-12 — No real-time file watching or indexing daemon

**Category:** time budget
**Boundary:** No `watchdog`/inotify loop, no debounce queue, no long-lived background indexer, no
"index is always fresh" claim. Freshness is an explicit user action.
**Instead we do:** `axiom reindex --to <rev>`, driven by `git diff --name-status <old>..<new>`
resolved into {A,M,D,R} (`FR-18`). Only A and M files are re-chunked and re-embedded; D chunks are
dropped from all three indexes; content-addressed blob reuse (`FR-19`) means an unchanged-content
rename costs **zero** embeddings. This is the correct shape for persona P-3 (CI/automation), which
wants a deterministic post-merge step, not a resident process. A watcher would be a wrapper around
this command, which is exactly why it is not needed to prove the capability.

### NG-13 — No hosted cloud deployment

**Category:** risk
**Boundary:** Nothing is deployed to a URL. No managed vector database (Pinecone, Weaviate Cloud,
Qdrant Cloud), no S3-backed index, no serverless function, no demo link that depends on our billing
account still working on 2026-09-27. A hosted demo is a single point of failure sitting outside our
control on judging day.
**Instead we do:** ship a Dockerfile and compose file that build the `axiom-retrieval` image from
`python:3.11-slim-bookworm` and run entirely on the evaluator's own machine, per
[Deployment.md](Deployment.md). After the model pre-download step in [Setup.md](Setup.md) §5.4 the
system runs fully offline with `HF_HUB_OFFLINE=1`. The assumption baked into the runbook is that
venue wifi fails.

### NG-14 — No cross-repo or federated search

**Category:** out of problem scope
**Boundary:** One repository per index root. No searching two unrelated codebases in one query, no
result merging across index roots, no remote index discovery, no monorepo-of-monorepos handling.
The problem statement describes one voice-assistant codebase.
**Instead we do:** multiple **versions of one repo**, which is a different axis and is the actual P1
goal. `.axiom/registry.json` maps `version_id` to manifest path and names the active version;
`--version <id>` scopes retrieval to one version (`FR-20`); `--all-versions` searches across all of
them and collapses near-duplicates into `SnippetFamily` records (`FR-21`). Federation across
repositories would add a dimension the goals never asked for.

### NG-15 — No code execution, sandboxing, or dynamic analysis

**Category:** risk
**Boundary:** Indexed code is never imported, evaluated, transpiled, bundled, instrumented, or run.
No `require()` of a demo-repo module to resolve a dynamic import, no test-suite execution to learn
call order, no runtime tracing. Executing arbitrary code from an indexed repository inside the
indexer is the single largest security hazard this design could have taken on, and it would also
make indexing non-deterministic.
**Instead we do:** purely static tree-sitter parsing into `structural.sqlite` (symbols, calls,
imports, exports). Call order comes from source order in `ChunkMetadata.calls`, not from a trace.
Where static analysis genuinely cannot resolve a target — dynamic dispatch, computed property
access, `eval` — we record nothing rather than guessing, and the query degrades to the dense and
sparse signals. Parse failure degrades to window chunking and never aborts the index (`NFR-07`).

### NG-16 — No security scanning of indexed code

**Category:** out of problem scope
**Boundary:** Axiom is not a SAST tool. No vulnerability detection, no CVE matching against
dependencies, no secret/credential scanning, no taint analysis, no license compliance. A query for
"where do we handle auth tokens?" returns the code; it does not audit it.
**Instead we do:** keep `FR-15`'s rule table narrowly about *performance smells on already-retrieved
code* — sync I/O in an async path, `await` inside a loop, unbounded `for..in` over a network
payload — and nothing else. Our own supply-chain and threat-model posture (pinned `uv.lock`, model
provenance, the unauthenticated-API trust boundary) is [Security.md](Security.md)'s subject, and it
is about Axiom, not about the corpus.

### NG-17 — No proprietary model API on the core path

**Category:** risk
**Boundary:** No OpenAI, Anthropic, Gemini, Cohere, or Voyage call is required for any scored path:
not for embedding, not for reranking, not for classification, not for expansion. Every weight is a
local file. A network-restricted judging laptop, an expired key, or a rate limit must not be able to
zero our score.
**Instead we do:** the locked local stack — `Qwen/Qwen3-Embedding-0.6B` INT8 with
`all-MiniLM-L6-v2` as fallback, `BAAI/bge-reranker-v2-m3` INT8 with `ms-marco-MiniLM-L-6-v2` as
fallback, and `Qwen2.5-1.5B-Instruct` Q4_K_M with a **heuristic rule engine** as fallback. The whole
pipeline is required to run green with `AXIOM_LLM_ENABLED=false` (`NFR-07`), which is also the
default posture for the eval run. The leaderboard context for this choice is in
[PRD.md](PRD.md#2-goals-and-success-metrics): the models above us on APPS are 3B/7B or proprietary
API tiers, and we are explicitly claiming a 0.6B CPU result instead of renting theirs.

### NG-18 — No ANN index on the benchmark corpus

**Category:** risk
**Boundary:** No IVF, no PQ, no HNSW, and no `nprobe` tuning for the CoIR `AppsRetrieval` run. The
corpus is **8,765 vectors**. FAISS `IndexFlatIP` over 8,765 normalised 1024-d vectors is exact and
effectively instant; any approximate structure can only *lose* recall in exchange for latency we do
not need, and `IndexIVFPQ` training below its recommended cluster population is a silent accuracy
leak that would be indistinguishable from a retrieval bug.
**Instead we do:** keep the locked switch — `IndexFlatIP` below 50k vectors, `IndexIVFPQ` at or
above 50k (`FR-05`). The benchmark sits firmly on the flat side; the 10k-chunk demo index also sits
on the flat side — and so, by a wide margin, does the demo index, now that the demo corpus is a
small tagged repo (≈10–50 source files, `OQ-07`) rather than the 10k-file repo an earlier revision
of `PRD.md §8` promised. `IndexIVFPQ` exists for scale headroom and is seeded when it is used
(`NFR-08`).
`VersionManifest.index_kind` records which was built so a result can never be misattributed.

### NG-19 — No HyDE in the default path

**Category:** CPU budget
**Boundary:** Hypothetical Document Embeddings stays behind `AXIOM_ENABLE_HYDE`, default **false**.
HyDE requires a generation pass before retrieval; on CPU with a Q4_K_M model that is hundreds of
milliseconds to seconds per query, and it contradicts `NG-23` in spirit by putting generated text at
the head of the pipeline.

**Correction, 2026-09-23.** The previous revision rejected HyDE by arithmetic — "hundreds of
milliseconds against 132 ms of headroom in the 768 ms p50 budget." That headroom figure was
fabricated (see `NG-10`), so a false number was driving a real scope decision. **The exclusion
stands on its own merits and is now stated without it:** HyDE adds a generation pass to the head of
the query path in a project whose entire thesis is that the LLM never generates on the retrieval
path, it is untested, it is unimplemented, and there are four calendar days. `AXIOM_ENABLE_HYDE` is
**closed for this cycle**, not deferred to a measurement that will not happen —
[`OQ-08`](OpenQuestions.md#oq-08--is-hyde-worth-its-latency-on-any-profile) is closed accordingly.

**Instead we do:** nothing at query time. The previous revision pointed at index-time reverse
doc2query expansion as the accepted alternative; that is also withdrawn (`NG-06`, ADR-013). The
accuracy work that remains is tuning the constants we already have, on the train split only
(`FR-26`), which is the honest and the affordable option.

### NG-20 — No custom React or SPA front-end

**Category:** time budget
**Boundary:** No React, Next.js, Vite, Tailwind, component library, or bundler. No dark-mode
toggle, no keyboard-shortcut palette, no animated transitions. Front-end build tooling is a
day-plus of work and zero rubric points.
**Instead we do:** Streamlit on port 8501 (`FR-25`): query box, query-type badge, result cards with
a syntax-highlighted snippet and a `file:line` header, per-signal rank breakdown, agent-pass
indicator, version selector, expandable snippet-family view. Under the cut order in
[PRD.md](PRD.md#71-cut-order-under-time-pressure) even this degrades to a minimal single-column
layout, because the CLI carries the demo.

### NG-21 — No Python 3.13 support

**Category:** risk
**Boundary:** 3.11 is the pinned primary and the CI/Docker interpreter; 3.12 is supported; 3.13 is
untested and unsupported. The native-wheel surface here is wide — `faiss-cpu`, `tree-sitter`,
`onnxruntime`, `llama-cpp-python` — and one missing wheel on 3.13 means a source build in the middle
of the build window.
**Instead we do:** `uv venv --python 3.11`, `requires-python = ">=3.11,<3.13"`, a committed
`uv.lock`, and a `python:3.11-slim-bookworm` base image, so the judge's environment and ours are the
same environment. [Setup.md](Setup.md) §7 documents the exact failure signature when a 3.13
interpreter sneaks in (setuptools attempting to compile a tree-sitter grammar).

### NG-22 — No generated replacement code for optimization hints

**Category:** out of problem scope
**Boundary:** This is the most easily misread line in the document, because optimization suggestions
*are* a stated bonus in the problem statement, so they are in scope — with a hard edge. The edge
sits exactly here:

| In scope (`FR-15`) | Out of scope (`NG-22`) |
|---|---|
| Detecting a static pattern in a chunk we already retrieved | Detecting anything in code we did not retrieve, i.e. whole-repo linting |
| Naming the issue from a fixed rule table: `await inside loop — consider Promise.all` | Writing the `Promise.all` version of the user's function |
| Populating `RetrievalResult.optimization_hint` with one fixed string per fired rule | Emitting a diff, patch, autofix, PR, or "here is the improved code" block |
| A hint the reader evaluates against the snippet shown directly above it | An LLM-authored prose rationale for the hint (that is `NG-03` and `NG-23`) |

**Instead we do:** a deterministic pattern-check pass over surfaced chunks only, keyed to three
rules at launch (sync I/O in an async path, `await` inside a loop, unbounded `for..in` over a
network payload), each mapping to one fixed string. `optimization_hint` is `str | None` and is
`None` for most results, which is the honest outcome. The moment a hint would need to be *written*
rather than *selected*, it is code generation and `NG-01` applies. `FR-15` is also first on the cut
list in [PRD.md](PRD.md#71-cut-order-under-time-pressure): it is a bonus on a bonus.

### NG-23 — No LLM ingestion of retrieved code

**Category:** risk
**Boundary:** The query LLM never receives code as input. Not a chunk, not a snippet, not a file, not
a summary of one. Its four permitted jobs are: classify the query, expand query terms, decompose the
query into sub-queries, and evaluate sufficiency. Sufficiency is judged from **scores**, not
content: the trigger is top-1 `rerank_score < 0.35` **or** fewer than 3 results above 0.20.
**Instead we do:** keep the agent loop bounded and code-blind — max 2 passes
(`AXIOM_AGENT_MAX_PASSES=2`) under a hard 5 s monotonic deadline (`FR-13`, `NFR-04`). This is the
direct answer to the constraint that the codebase is larger than any context window: if the LLM
never reads code, corpus size cannot degrade the agent, and the same loop behaves identically on a
10-file repo and a 10,000-file repo. It is also what keeps `AXIOM_LLM_ENABLED=false` a first-class
mode rather than a crippled one — with the LLM off, the heuristic rule engine does the same four
jobs and retrieval quality moves, but nothing breaks.

### NG-24 — No non-git version sources

**Category:** time budget
**Boundary:** Version identity comes from git. No SVN, no Mercurial, no tarball-snapshot diffing, no
mtime-based change detection, no directory-comparison heuristic to reconstruct history. If the
target is not a git repository, there is no incremental path.
**Instead we do:** `git diff --name-status <old>..<new>` as the single source of change truth
(`FR-18`), with `version_id`, `commit_sha`, and `last_modified` stamped onto every chunk at index
time (`FR-16`). A non-repo target supplies `version_id` through config and gets a full index — which
is correct, not degraded, because there is nothing to diff against. `VersionManifest.file_hashes`
gives a content-level cross-check so a broken `git diff` (for example the `core.autocrlf` trap in
[Setup.md](Setup.md) §7) is detectable rather than silent.

### NG-25 — No learning-to-rank or click feedback

**Category:** risk
**Boundary:** No LambdaMART, no XGBoost ranker over signal features, no implicit feedback from UI
clicks, no online weight updates, no bandit over profiles. Ranking behaviour is fixed at config
time and identical for every user and every run.
**Instead we do:** weighted RRF with weights that are *fitted offline and committed*:
`score(d) = Σ_i w_i / (60 + rank_i(d))`, weights read from `QueryPlan.strategy_weights` and tuned on
the train split per `FR-26`. A learned ranker would break `NFR-08` determinism, make the ablation
table (dense only / +sparse / +structural / +rerank / +agent) uninterpretable, and require labelled
interaction data that does not exist for a 10-day project.

### NG-26 — No indexing of build output, vendored, or non-source assets

**Category:** CPU budget
**Boundary:** `node_modules/`, `dist/`, `build/`, `coverage/`, `.min.js`, source maps, lockfiles,
images, and binary blobs are never chunked or embedded. On a real JavaScript repo these dominate
byte count while carrying near-zero retrieval value, and a single minified bundle can produce
thousands of junk chunks that poison both BM25 term statistics and the dense neighbourhood.
**Instead we do:** an explicit ignore list in the profile configs, applied before chunking, plus the
64-512 token chunk target with sub-16-token merging (`FR-04`) so pathological inputs cannot inflate
chunk counts. This fence is a direct dependency of *any* cold-index figure `NG-07`/`NFR-01` ever
carries: a chunk count is only meaningful over a source-only corpus. (The 636 s figure this
paragraph previously cited is retracted — see `NG-07`.)

### NG-27 — No graph database for the structural index

**Category:** time budget
**Boundary:** No Neo4j, no ArangoDB, no long-lived graph service, no Cypher. Adding a second
datastore means a second container, a second connection lifecycle, a second failure mode on the
judge's laptop, and a second thing to explain in five minutes.
**Instead we do:** `structural.sqlite` with four relations — symbols (name, kind, file, line span,
scope), calls (caller, callee, source order), imports (file, module), exports (module, symbol) —
queried with plain SQL for callers-of, callees-of, imports-of, exports-of, and ordered-call-pair
traversal (`FR-09`, `FR-10`). The DDL is locked in the contract and mirrored in
[Schema.md](Schema.md). One file, zero daemons, and the ordered-call-pair query that answers query
archetype Q2 is a self-join with an index on `(caller, source_order)`.

### NG-28 — No native Windows-arm64 support

**Category:** risk
**Boundary:** `faiss-cpu` publishes no win-arm64 wheel, and a source build of FAISS during the build
window is an unbounded time sink. We will not attempt one, and we will not ship a FAISS-free code
path to work around it.
**Instead we do:** the documented WSL2 route in [Setup.md](Setup.md) §2.3 — Ubuntu 24.04 under WSL2,
repo kept inside the WSL filesystem (`~/axiom`, never `/mnt/c`, which makes index writes roughly 5x
slower and breaks `git diff` mtime assumptions). Windows x86_64 and macOS arm64 remain supported dev
targets; Linux x86_64 is the CI and judge reference. Path handling is platform-independent
regardless: `ChunkLocation.file_path` is always POSIX-separated and repo-relative.

### NG-29 — No tuning on the benchmark test split

**Category:** risk
**Boundary:** The `AppsRetrieval` **test** split is never used to select a weight, a threshold, a
model, a chunk size, or a prompt. Not once, not "just to check", not by eyeballing per-query
failures. Every such use overfits a number we then report as a generalisation claim, and a jury that
asks how we picked 0.85/0.15 deserves an answer that is not "we tried it on the test set".
**Instead we do:** tune exclusively on the 5,000 train pairs, split 4,000 tune / 1,000 dev with a
committed seeded id list under `data/splits/` (`FR-26`). Each candidate configuration is scored on
test **at most once**, and every test-split run — date, config hash, profile, resulting NDCG@10 and
MRR — is logged in [Tracker.md](Tracker.md). The protocol itself is [TestPlan.md](TestPlan.md)'s
subject. This is the one non-goal whose violation is invisible in the artefact and fatal to the
claim, which is why it is written down rather than assumed.

**Absolute, confirmed 2026-09-23 — including under time pressure.** The fence has no fallback door,
and it previously had one. `PRD.md` Assumption A-6's fallback read: "if the train split is
unavailable or mis-shaped, fall back to a 500-query holdout **carved from the dev portion of
test**." That is precisely this non-goal's violation, written into a requirements document as the
contingency plan. A-6 has been rewritten
([PRD.md §10](PRD.md#10-assumptions)) so that **no fallback path can reach the test split**: if the
train split fails, we tune on the demo corpus with a hand-written query set, or we ship the locked
default constants untuned and say so. A lower reported number that is honestly obtained is
recoverable; a number obtained this way is not, and it cannot be detected from the artefact by
anyone but us. `PROJECT_OVERVIEW.md`'s Day-7 "MTEB eval on test split" entry, if it is retained at
all, means the single permitted reportable run, never a tuning activity.

### NG-30 — No cross-query result cache

**Category:** risk
**Boundary:** No memoisation of `(query, version) -> results`, no warm result store, no
pre-computation of demo queries. A cache would make our own latency measurements meaningless, hide
regressions behind a hit rate, and let a demo look fast for reasons unrelated to the architecture.
**Instead we do:** cache exactly one thing, at exactly one layer — content-addressed **embeddings**
at `.axiom/blobs/<content_hash>.npy`, shared across versions (`FR-19`). That is an indexing-time
artefact with a content-derived key, so it cannot go stale. Query-time cost is paid every time and
reported every time: whatever p50 `NFR-03` eventually measures is a cold-cache-by-construction
number. Model load
(ONNX session construction, GGUF mmap) is lazy and therefore warm after the first query; the demo
runbook in [Deployment.md](Deployment.md) warms the process explicitly rather than pretending the
first query was fast.

---

## 4. Deferred, not rejected

These are good ideas. Each is excluded only because it does not fit between 2026-09-15 and
2026-09-27, and each would be a reasonable first item after submission. Nothing here is promised,
and nothing here may be started before the `Definition of Done` in
[PRD.md](PRD.md#21-definition-of-done) is fully green.

| Idea | Why it is outside the 10-day window |
|---|---|
| TypeScript and JSX grammars for the structural signal | Each new grammar needs its own node-kind mapping, export-resolution rules, and chunker tests; that is a day per language and the demo repo is plain JavaScript. |
| Learned sparse retrieval (SPLADE-style) as a fourth signal | Requires an expansion model pass over the whole corpus at index time, on top of a cold index already projected to miss the 720 s ceiling unaided (`OQ-14`). |
| Embedding fine-tune on APPS train pairs | Explicitly rejected for this cycle by `NG-06`; revisiting it needs GPU hours and a held-out protocol that does not exist yet. |
| VS Code extension over the existing `--json` API | Extension packaging and an editor debug loop are a day-plus for zero rubric points; `NG-11` already froze the contract it would consume. |
| File-watching daemon wrapping `axiom reindex` | Pure convenience over a capability `FR-18` already proves; debounce and partial-write handling are where the real time goes. |
| Call-graph reachability beyond depth 1 (transitive callers) | Needs a recursive CTE plus cycle handling and a new relevance story for indirect hits; the three query archetypes are all depth-1. |
| Multi-hop agent planning beyond 2 **total** passes (§5.3 of TechSpecifications: the cap counts the initial retrieval) | The 5 s wall clock in `FR-13` cannot fund a third cycle on CPU; raising the cap needs a faster reranker first. |
| Cross-encoder distillation into a smaller student | Would buy latency headroom for HyDE (`NG-19`), but it is training, which `NG-06` excludes this cycle. |
| Hosted read-only demo instance | Useful for sharing after judging; a live URL on judging day is a `NG-13` single point of failure. |
| Prometheus/OpenTelemetry export of the `timings` block | `NFR-10` already emits the data structurally; wiring an exporter and a dashboard serves an operations story `NG-10` says we do not have. |
| Query-log-driven evaluation set from real developer usage | There are no real users during the build window, so there is no log to mine; the 20 hand-labelled demo queries in [TestPlan.md](TestPlan.md) stand in. |
| Snippet-family visual diff viewer in the UI | `FR-21` already computes `diffs`; rendering them as an interactive three-way view is polish that competes with the cut list. |

---

## 5. Changing this fence

| Situation | Correct action |
|---|---|
| A jury-relevant requirement turns out to need something listed here | Delete the `NG-##` in a PR approved by all four members, add an `ADR-###` in [Decisions.md](Decisions.md) naming what was cut to pay for it, and add the `FR-##` to [PRD.md](PRD.md). |
| You are unsure whether something is excluded | It is excluded. Ask in the daily sync; do not build it speculatively. |
| The exclusion is settled but the *reason* is still being argued | That is an `OQ-##` in [OpenQuestions.md](OpenQuestions.md), not a licence to build. `OQ-01` (two-profile split) and `OQ-02` (sparse weight on the eval profile) are the two that touch this document most directly. |
| A non-goal turns out to be already half-built | Delete the code. A half-built excluded feature is the worst outcome available: it costs review time, breaks `NFR-11`, and earns nothing. |

IDs are permanent. A deleted non-goal leaves its `NG-##` retired with a one-line note and a pointer
to the `ADR-###` that retired it; numbers are never reused, because other documents cite them by
number.
