# Security

Threat model, trust boundaries, data handling, and dependency security for PRISM.

**Owner:** Harshdeep Athawale
**Last updated:** 2026-09-16
**Status:** Draft

Related: [NonGoals.md](NonGoals.md) · [Rules.md](Rules.md) · [Schema.md](Schema.md) · [TestPlan.md](TestPlan.md) · [Deployment.md](Deployment.md) · [_CONTRACT.md](_CONTRACT.md) · [API.md](API.md)

---

## 1. Scope and posture

Axiom is a hackathon prototype that indexes and retrieves from a local codebase. It is not a
multi-tenant service, has no user accounts, and is never deployed anywhere but the operator's own
machine — see [`NG-08`](NonGoals.md#ng-08--no-authentication-authorisation-or-multi-tenancy),
[`NG-13`](NonGoals.md#ng-13--no-hosted-cloud-deployment). This document does not re-argue those
exclusions; it states what security posture follows from them, and it covers the parts of the attack
surface that are real regardless of deployment posture: what Axiom persists about the code it reads,
what runs unauthenticated on a local port, and what enters the process from the outside world (model
downloads, user queries, an indexed repository's own content).

The single governing fact: **Axiom is a passive reader.** It parses, embeds, and indexes text. It
never executes, transpiles, or evaluates a byte of the code it indexes
([`NG-15`](NonGoals.md#ng-15--no-code-execution-sandboxing-or-dynamic-analysis)), and it is not a
security scanner of that code either
([`NG-16`](NonGoals.md#ng-16--no-security-scanning-of-indexed-code)). Most of what a "security
review" would normally ask about a code-processing tool — sandbox escapes, injection through
executed input — does not apply here, precisely because execution is the one thing this system
structurally cannot do to its input. What remains is the surface below.

---

## 2. Threat model — STRIDE

Applied to Axiom's actual components: the CLI (local filesystem access), the FastAPI dev server
(`FR-24`), the Streamlit UI (`FR-25`), the indexed corpus, and the model supply chain (Hugging Face
Hub downloads).

### 2.1 Spoofing

| Applies? | Detail |
|---|---|
| To the API/UI | **No.** There is no identity to spoof — no accounts, no sessions, no auth tokens ([`NG-08`](NonGoals.md#ng-08--no-authentication-authorisation-or-multi-tenancy), [`NG-09`](NonGoals.md#ng-09--no-persistent-user-accounts-history-or-personalisation)). Anyone who can reach the loopback-bound port *is* the operator, by construction — there is no second identity to impersonate. |
| To the model supply chain | **Partially.** A malicious actor controlling DNS or a MITM position between the operator and `huggingface.co` could theoretically serve a spoofed model artefact. Mitigated by HF Hub's own TLS and content-addressed blob storage (the hub keys model files by content hash internally), and by pinning model revisions rather than `main`/`latest` (§4). We do not add our own signature verification layer — out of scope for a 10-day project, and redundant with the hub's own integrity guarantees for this threat's realistic likelihood. |
| To version identity | **No.** `version_id`/`commit_sha` are read from `git`, a locally-trusted source in this threat model — we do not defend against a compromised local git installation. |

### 2.2 Tampering

| Applies? | Detail |
|---|---|
| To `.axiom/` index artefacts | **Yes, in principle** — nothing cryptographically signs `chunks.jsonl`, `dense.faiss`, or `structural.sqlite`. `Chunk`'s `_ids_match_content` validator ([Schema.md §6](Schema.md#6-chunk)) *does* catch accidental corruption or hand-editing on load — recomputing `chunk_id` and `content_hash` and rejecting a mismatch — but this is a data-integrity check, not a security control against a deliberately tampered index that also forges consistent hashes. Accepted risk: an attacker with local write access to `.axiom/` already has write access to the source repository it was built from, so tampering with the index grants no privilege the attacker did not already have. |
| To the indexed source repository | **Out of scope by construction.** Axiom reads whatever is on disk at index time; it has no opinion on whether that content was itself tampered with upstream (a supply-chain question about *the target repo*, not about Axiom). |
| To in-flight query/response traffic | **No transport security by design.** The API is loopback-only HTTP, not HTTPS ([`NG-08`](NonGoals.md#ng-08--no-authentication-authorisation-or-multi-tenancy)'s dev-only posture). Adding TLS to a loopback-bound dev server defends against a threat model (a network attacker on the loopback interface) that requires the attacker to already have code execution on the same machine, at which point TLS on `localhost:8000` is not the control that stops them. |

### 2.3 Repudiation

| Applies? | Detail |
|---|---|
| Query audit trail | **Not a security control, but present for a different reason.** [`NFR-10`](PRD.md#6-non-functional-requirements) requires every stage to emit a structured timing record and every `--json` response to carry a `timings` block. This exists for latency auditability (so a performance claim is checkable), not for non-repudiation — there is no identity attached to a query to repudiate ([`NG-09`](NonGoals.md#ng-09--no-persistent-user-accounts-history-or-personalisation)). We do not build an access log keyed to a user identity, because no user identity exists. |

### 2.4 Information Disclosure

This is the category that matters most, and it is not about Axiom's own code — it is about what
Axiom **persists about the corpus it was pointed at.** See §3.

| Applies? | Detail |
|---|---|
| Indexed source code, verbatim, in `.axiom/` | **Yes — this is the primary information-disclosure surface.** Treated at length in §3. |
| Model weights, HF cache | **No new disclosure** — these are public model repositories, not secrets. |
| Error messages | **Bounded.** [Rules.md §9.2](Rules.md#92-error-taxonomy) requires exception messages to name "the offending value and the config field that controls it" — deliberately informative for debugging, which means an error response can echo back a malformed query or a bad config value. It never echoes chunk content or file-system paths outside the configured index root, and `TC-071` ([TestPlan.md](TestPlan.md#38-category-h--end-to-end-pipeline-cli-api-ui-tc-065--tc-072)) explicitly asserts the API "no stack trace in the body" for malformed/hostile input. |
| Path traversal via API parameters | **Tested, not assumed safe.** `TC-071` posts `version="../../etc"` and other hostile input and asserts no filesystem access is attempted, patched at the `open` call site during the test. The `version` parameter is resolved through `registry.json`'s known version set, never concatenated directly into a filesystem path. |

### 2.5 Denial of Service

| Applies? | Detail |
|---|---|
| Adversarial query input hanging or crashing the process | **Directly addressed by [Rules.md Rule 3](Rules.md#rule-3--never-raise-on-bad-input-degrade).** Every stage has a declared degradation ladder; nothing raises on malformed input. `TC-047` specifically tests ReDoS resistance in the regex fallback tokenizer against 12 adversarial inputs (`"a"*50000`, nested-quantifier-shaped strings) with a hard timeout. `TC-086` proves the agent loop terminates within its wall-clock budget even under a scripted-always-insufficient adversarial condition. |
| Resource exhaustion via a very large query or a very large indexed file | **Bounded.** Query length is truncated to `settings.max_query_chars` (`TC-010`). A single oversized source file degrades through the chunking ladder (statement-boundary split → line-window fallback) rather than blowing memory (`TC-023`'s 1.2 MB minified-bundle case, `TC-024`'s 200-deep-nesting case). |
| Traffic-volume DoS against the API | **Not defended, and not in scope.** There is no rate limiter ([`NG-08`](NonGoals.md#ng-08--no-authentication-authorisation-or-multi-tenancy)). A loopback-bound dev server with no external network exposure has no realistic traffic-volume DoS threat actor in this deployment model — the only process that can reach it is a process already running on the same machine. |

### 2.6 Elevation of Privilege

| Applies? | Detail |
|---|---|
| API/UI granting access beyond what local filesystem access already grants | **No — this is the exact reasoning [`NG-08`](NonGoals.md#ng-08--no-authentication-authorisation-or-multi-tenancy) states directly:** "anyone who can reach port 8000 can read the index, and anyone who can read the index can already read the repo it was built from — so the API grants no privilege that the filesystem did not already grant." There is no privilege boundary inside Axiom for an attacker to escalate across, because there is only one privilege level to begin with. |
| Code execution via the indexed corpus | **Structurally impossible**, not merely mitigated — [`NG-15`](NonGoals.md#ng-15--no-code-execution-sandboxing-or-dynamic-analysis) means there is no `eval`, `require()`, transpile, or test-run path anywhere in the indexing or retrieval pipeline. A maximally adversarial JavaScript file in the indexed corpus is, to Axiom, inert text to be parsed by tree-sitter (a pure parser, not an interpreter) and embedded by a model that has no execution semantics. |
| Model-loading code execution (a malicious model artefact) | **Bounded by format choice and scope.** ONNX Runtime and `llama-cpp-python` load structured weight formats, not arbitrary pickled Python objects — unlike a raw PyTorch `.pt`/`.bin` load via `torch.load` with `weights_only=False`, which is a known code-execution vector. Axiom's model loading path uses ONNX (`.onnx`) and GGUF, both data formats without an embedded-code-execution primitive. |

---

## 3. Trust boundaries

```
                    ┌─────────────────────────────────────────────┐
                    │              INSIDE THE BOUNDARY              │
                    │                                                │
   query text  ───▶ │   axiom process (CLI / API / UI)              │
                    │        │                                       │
                    │        ├── .axiom/ index (chunks, vectors,    │
                    │        │   structural.sqlite) — see §3.1       │
                    │        │                                       │
                    │        └── model cache (HF_HOME)               │
                    │                                                │
   RetrievalResult  │                                                │
   JSON       ◀──── │                                                │
                    └─────────────────────────────────────────────┘
                                    ▲                    ▲
                                    │                    │
                          OUTSIDE the boundary   OUTSIDE the boundary
                    ┌──────────────────────┐   ┌─────────────────────────┐
                    │ the network (loopback │   │ the indexed repository's│
                    │ only; no other process│   │ own execution — NEVER   │
                    │ than the operator's   │   │ run, per NG-15. Axiom   │
                    │ own has any reason to │   │ reads it once, at index │
                    │ reach 127.0.0.1:8000) │   │ time, as inert text.    │
                    └──────────────────────┘   └─────────────────────────┘
```

The boundary is the `axiom` process itself. Exactly two things legitimately cross it:

| Crosses the boundary | Direction | Contains |
|---|---|---|
| Query text | In | A string, plus optional `top_k`/`version`/`--all-versions` parameters. Bounded per §2.5. |
| `RetrievalResult` JSON | Out | `chunk.text` (verbatim source), `chunk.location`, `score`, `match_reason`, `signals`, optional `optimization_hint` — see [Schema.md §9](Schema.md#9-retrievalresult). This is, by design, a **copy of the indexed source code** flowing back out to whoever queried it. |

That second row is the actual security-relevant fact about this system: **Axiom's entire purpose is
to disclose (to the operator) the exact content of the corpus it indexed.** This is not a flaw to
mitigate — it is the product — but it means the trust boundary around `.axiom/` and around the API
must be treated as equivalent to the trust boundary around the source repository itself, not as a
lesser one. Anyone the operator would not hand the raw source repository to should not be given
network access to the API or a copy of the `.axiom/` directory either, per [`NG-08`](NonGoals.md#ng-08--no-authentication-authorisation-or-multi-tenancy)'s
own reasoning.

### 3.1 The `.axiom/` index carries the same sensitivity as the source repository

Per the on-disk layout locked in `_CONTRACT.md §6`:

```
.axiom/
├── registry.json
├── blobs/<content_hash>.npy          # embedding vectors — NOT reversible to source text
└── index/<version_id>/
    ├── manifest.json
    ├── chunks.jsonl                  # VERBATIM SOURCE TEXT, one Chunk per line
    ├── dense.faiss
    ├── dense.idmap.json
    ├── sparse.bm25s/                 # BM25 postings — derived from source tokens
    └── structural.sqlite             # symbol names, call names, import paths
```

`chunks.jsonl` is the concrete disclosure surface: it contains every indexed chunk's `text` field
byte-for-byte, per [Schema.md §6](Schema.md#6-chunk) ("Verbatim source text of the chunk"). If the
indexed repository contains a secret accidentally committed by *its own* history — an API key left
in a comment, a hardcoded credential in a test fixture — that secret is now duplicated into
`chunks.jsonl`, into the BM25 postings, and (irreversibly one-way, but present) contributes to an
embedding vector in `blobs/`. Axiom does not scan for this
([`NG-16`](NonGoals.md#ng-16--no-security-scanning-of-indexed-code) is explicit that secret-scanning
is out of scope), and it does not redact it — a redaction pass would violate the byte-exact
round-trip guarantee `NG-01` depends on for the deliverable itself.

**The existing control, and it is sufficient for this threat model:** [Rules.md §9.5](Rules.md#95-what-may-and-may-not-be-committed)
already forbids committing any `.axiom/` artefact to git — `dense.faiss`, `structural.sqlite`,
`*.npy`, `chunks.jsonl`, and the whole tree are in the "Never commit" table, enforced by `.gitignore`
as the executable form of that rule. This means the disclosure surface never leaves the machine it
was built on by accident via version control, which is the realistic failure mode for a hackathon
project (someone `git add -A`-ing a working directory). It does not mean the directory is safe to
casually copy, screen-share, or upload elsewhere — **inherit whatever access-control posture the
source repository itself required**, and communicate that plainly to anyone handed a copy of
`.axiom/` for debugging or demo purposes.

### 3.2 What is explicitly never inside the boundary

- **The indexed repository's own runtime** — never started, never imported, never transpiled
  ([`NG-15`](NonGoals.md#ng-15--no-code-execution-sandboxing-or-dynamic-analysis)).
- **Any proprietary model API** — no network call to an external inference provider on the scored
  path ([`NG-17`](NonGoals.md#ng-17--no-proprietary-model-api-on-the-core-path)), so there is no
  third party the query text or retrieved code is ever transmitted to.
- **Any other process on the host, by default** — the API binds `127.0.0.1`, not `0.0.0.0`, outside
  of the explicit Docker networking case in [Deployment.md §3](Deployment.md#3-docker), where the
  container boundary itself is the isolation mechanism.

---

## 4. Data handling

| Question | Answer |
|---|---|
| Does Axiom persist the corpus it indexes? | Yes — verbatim, in `chunks.jsonl` and derivatively in the sparse and dense indexes. See §3.1. |
| Does Axiom transmit the corpus anywhere? | No, beyond the loopback API response to the operator's own query — see §3, "crosses the boundary" table. Never to a third-party model API (§3.2). |
| Does Axiom scan the corpus for secrets or vulnerabilities? | No — explicitly out of scope, [`NG-16`](NonGoals.md#ng-16--no-security-scanning-of-indexed-code). |
| Does Axiom execute the corpus? | No — structurally impossible, [`NG-15`](NonGoals.md#ng-15--no-code-execution-sandboxing-or-dynamic-analysis). |
| Does Axiom retain query text or results across sessions? | No — [`NG-09`](NonGoals.md#ng-09--no-persistent-user-accounts-history-or-personalisation): no query history, no saved searches, no telemetry upload. A query leaves no trace beyond a local, unshipped log line. |
| What is the operator's responsibility? | Treat `.axiom/` for a given corpus with the same access-control discipline as the corpus itself (§3.1), and never commit it to version control (already enforced by `.gitignore` per [Rules.md §9.5](Rules.md#95-what-may-and-may-not-be-committed)). |
| Is the CoIR benchmark corpus sensitive? | No — it is a public dataset (`CoIR-Retrieval/apps` on the HF Hub); the data-handling concerns in this section are about **operator-supplied corpora** (the live-demo repo, or any real repository a future user points Axiom at), not the benchmark. |

---

## 5. Dependency security

Two supply chains: Python packages, and model weights. Both are pinned, for the same reason —
[Rules.md §7](Rules.md#7-configuration-discipline)'s "a reported score must be reproducible from a
git SHA alone" is a reproducibility argument that is also, incidentally, a supply-chain security
argument: a floating dependency version is both a reproducibility risk and an unreviewed-code
risk every time it moves.

### 5.1 Python packages

Per [Rules.md §9.4](Rules.md#94-dependency-pinning):

- `uv.lock` is committed and authoritative; `requirements.txt` is generated from it, never
  hand-written, for the `pip`-fallback install path.
- Direct dependencies are pinned to an **exact** version in `pyproject.toml`
  (`faiss-cpu==1.8.0`-style), not a range — a hackathon-scale project has no appetite for a surprise
  minor-version bump changing behaviour on 24 September.
- Adding a dependency requires, in the same PR: what it replaces or enables, a licence check
  (**permissive only** — MIT/BSD/Apache-2.0), an install-size note, and confirmation it has no
  GPU-only wheel on `linux/amd64`.
- No dependency may be added after the Day-9 feature freeze except to fix a submission-blocking
  defect ([Rules.md §9.4](Rules.md#94-dependency-pinning) item 4).

**CI enforcement:** [TestPlan.md §8](TestPlan.md#8-ci-pipeline) stage 5 runs
`pip-audit -r requirements.txt --strict`, blocking merge. This is the mechanical check that a pinned
version has not since been disclosed as vulnerable — pinning alone only guarantees reproducibility,
not that the pinned version is still considered safe, which is exactly what `pip-audit` verifies on
every CI run rather than once at pin time.

### 5.2 Model weight provenance

A distinct supply chain from Python packages, and handled distinctly:
[Rules.md §9.4](Rules.md#94-dependency-pinning) item 5 requires model weights be "pulled at runtime
into a gitignored cache directory, pinned by revision hash where the hub supports it. Never
`main`/`latest`." This matters because a model repository on the Hub can be updated by its owner at
any time — pinning to a named revision (not a mutable branch pointer) is what makes "the model we
tested against" and "the model that ships" provably the same artefact, the same reproducibility
argument as `uv.lock` applied to weights instead of packages.

| Model | Repo | Pinning discipline |
|---|---|---|
| Dense embedder | `Qwen/Qwen3-Embedding-0.6B` | Revision pinned in `configs/*.yaml`, not `main` |
| Cross-encoder reranker | `BAAI/bge-reranker-v2-m3` | Revision pinned |
| Query LLM | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` | Specific GGUF filename pinned (`qwen2.5-1.5b-instruct-q4_k_m.gguf`), not a directory glob |
| Fallback embedder / reranker | `all-MiniLM-L6-v2`, `ms-marco-MiniLM-L-6-v2` | Same discipline, since a fallback that silently drifted would defeat `NFR-07`'s determinism-under-degradation guarantee |

We do not additionally verify a cryptographic signature over the downloaded weight files (§2.1) —
the Hub's own content-addressed storage and TLS transport are treated as sufficient integrity
controls for this project's realistic threat model, and building an independent signature-checking
layer is out of proportion to a 10-day hackathon submission.

---

## 6. What we deliberately do not do

A security-framed summary of the relevant [NonGoals.md](NonGoals.md) entries — this table exists so
a reviewer can see the whole security-relevant scope-fence in one place; the full rationale for each
lives at its `NG-##`, not repeated here.

| We do not... | Security rationale, in one sentence | Full non-goal |
|---|---|---|
| Authenticate API/UI access | There is exactly one privilege level (the local operator); an auth layer would protect a boundary that does not exist | [`NG-08`](NonGoals.md#ng-08--no-authentication-authorisation-or-multi-tenancy) |
| Execute, sandbox, or dynamically analyse indexed code | Removes the entire code-execution attack class from the indexed corpus's threat surface structurally, not by policy | [`NG-15`](NonGoals.md#ng-15--no-code-execution-sandboxing-or-dynamic-analysis) |
| Scan the indexed corpus for secrets or vulnerabilities | Out of product scope; scanning a corpus for secrets and then persisting the scan results would itself be a new disclosure surface to reason about | [`NG-16`](NonGoals.md#ng-16--no-security-scanning-of-indexed-code) |
| Depend on a proprietary model API for any scored path | No query text or retrieved code is ever transmitted to a third party; a network-restricted judging machine cannot be zeroed by an expired key or rate limit | [`NG-17`](NonGoals.md#ng-17--no-proprietary-model-api-on-the-core-path) |
| Host a deployment anywhere but the operator's own machine | No externally-reachable attack surface exists at all outside a `docker compose up` the operator themselves ran | [`NG-13`](NonGoals.md#ng-13--no-hosted-cloud-deployment) |

---

## 7. Related documents

| Document | Relationship |
|---|---|
| [NonGoals.md](NonGoals.md) | The scope fence this document's threat model is built against |
| [Rules.md](Rules.md) | §9.4 dependency pinning, §9.5 what may not be committed — the two controls §3.1/§5 rely on |
| [Schema.md](Schema.md) | `Chunk.text`'s verbatim-content guarantee, the disclosure surface described in §3.1 |
| [TestPlan.md](TestPlan.md) | `TC-047` ReDoS test, `TC-071` API hostile-input test, §8 CI `pip-audit` stage |
| [Deployment.md](Deployment.md) | The loopback-only, unauthenticated posture this document's trust-boundary section assumes |
| [API.md](API.md) | The concrete request/response shapes crossing the boundary described in §3 |
