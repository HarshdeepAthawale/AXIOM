# Axiom

Agentic code intelligence: multi-signal, version-aware code retrieval that runs entirely on CPU.

**Owner:** Harshdeep Athawale
**Last updated:** 2026-09-23
**Status:** Draft

> **On the name.** The project is **Axiom** (`ADR-015`, Accepted): the package, the import root, the
> CLI (`axiom`), the environment prefix (`AXIOM_`) and the index directory (`.axiom/`) all carry it.
> **PRISM** appears in exactly two places and refers to the Samsung programme, not this project: the
> event name, "Samsung PRISM GenAI Hackathon", and the organiser-prescribed release tag
> `PRISM_GENAI_HACKATHON_Y2026`.

---

## What Axiom is

A multi-pass agentic code retrieval system. Given a natural-language query and a
code library too large for any LLM context window, it returns a **ranking of code
snippets with file and line locations**. Retrieval only — no code generation, no
answer synthesis.

Three retrieval signals — dense embeddings, sparse BM25, and structural AST/call-graph —
fused by Reciprocal Rank Fusion, refined by a cross-encoder reranker, and driven by a
bounded agentic loop that classifies, plans, evaluates and rewrites. CPU-only.

| | |
|---|---|
| Event | Samsung PRISM GenAI Hackathon 3rd Edition (2026-27), Theme 01 |
| Team | Incognito |
| Build window | 2026-09-11 → 2026-09-27 (Day 10 = 2026-09-24) |
| Release tag | `PRISM_GENAI_HACKATHON_Y2026` |

**Start here:** [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) — the problem, the architecture, the
component walkthrough, the measured results and the jury-scoring breakdown, in one document.

---

## Quickstart

Five commands from a clean clone to ranked results, CPU-only. Full detail, per-platform notes and
the verification ladder are in [Setup.md](docs/Setup.md).

```bash
git clone https://github.com/HarshdeepAthawale/Samsung-Prism-Hack.git && cd Samsung-Prism-Hack
uv venv --python 3.11 && source .venv/bin/activate     # Windows: .venv\Scripts\activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU torch FIRST
uv sync --frozen
axiom index tests/fixtures/mini_repo --version-id smoke --index-root /tmp/axiom-smoke
axiom query "how is user input normalized before dispatch" --version smoke --index-root /tmp/axiom-smoke --top-k 3
```

The subcommand is `query`. There is no `axiom search`. Container path:
`docker compose up --build`, then `curl -s http://127.0.0.1:8000/v1/health` — see
[Deployment.md](docs/Deployment.md).

---

## Read in this order

New to the project? Follow this path.

1. [PRD.md](docs/PRD.md) — what we are building and why; requirements and success metrics
2. [Design.md](docs/Design.md) — architecture, diagrams, and the reasoning behind the shape
3. [Schema.md](docs/Schema.md) — the shared data model; **all four workstreams code against this**
4. [TechSpecifications.md](docs/TechSpecifications.md) — component-by-component engineering spec
5. [Setup.md](docs/Setup.md) — get it running from a clean clone
6. [Rules.md](docs/Rules.md) — the binding engineering invariants; read before your first commit

---

## Full index

### Core

| Doc | Purpose | Owner |
|---|---|---|
| [PRD.md](docs/PRD.md) | Problem, personas, `FR-##` / `NFR-##` requirements, success metrics, jury alignment | Parth |
| [TechSpecifications.md](docs/TechSpecifications.md) | Runtime, model stack, per-module specs, algorithms, config reference | Prabinder |
| [Appflow.md](docs/Appflow.md) | Eight end-to-end runtime flows with sequence diagrams and budgets | Harshdeep |
| [Design.md](docs/Design.md) | Architecture, principles, concurrency model, degradation ladder, extension points | Anish |
| [Schema.md](docs/Schema.md) | Pydantic data model, id scheme, on-disk formats, SQLite DDL, invariants | Prabinder |
| [ImplementationPlan.md](docs/ImplementationPlan.md) | Workstreams, day-by-day plan, `M0`–`M7` gates, `RISK-01`–`RISK-12` | Prabinder |
| [Tracker.md](docs/Tracker.md) | `T-###` task board, burndown, standup log, eval metrics log | Parth |
| [Rules.md](docs/Rules.md) | Cardinal rules, engineering invariants, anti-pattern gallery | Harshdeep |
| [_CONTRACT.md](docs/_CONTRACT.md) | The locked technical contract every other doc defers to | Prabinder |

### Decisions and history

| Doc | Purpose | Owner |
|---|---|---|
| [Decisions.md](docs/Decisions.md) | `ADR-###` log: what was chosen, why, alternatives rejected | Prabinder |
| [Changelog.md](docs/Changelog.md) | What shipped and when; breaking vs non-breaking; planned releases | Parth |

### Setup and ops

| Doc | Purpose | Owner |
|---|---|---|
| [Setup.md](docs/Setup.md) | Prerequisites, install paths, env vars, verification ladder, troubleshooting | Parth |
| [Deployment.md](docs/Deployment.md) | Docker, submission runbook, rollback, demo-day runbook | Parth |
| [Submission.md](docs/Submission.md) | PPT outline, demo-video script, demo-day runbook, the results table as it goes on the slide | Harshdeep |

### Quality and safety

| Doc | Purpose | Owner |
|---|---|---|
| [TestPlan.md](docs/TestPlan.md) | `TC-###` cases, edge catalogue, performance tests, eval protocol, CI | Parth |
| [Security.md](docs/Security.md) | Threat model (STRIDE), trust boundaries, data handling, dependency security | Harshdeep |

### API and integration

| Doc | Purpose | Owner |
|---|---|---|
| [API.md](docs/API.md) | HTTP endpoint contracts, error codes, CLI reference | Harshdeep |

### Scope control

| Doc | Purpose | Owner |
|---|---|---|
| [NonGoals.md](docs/NonGoals.md) | `NG-##` explicitly out-of-scope items and what we do instead | Parth |
| [OpenQuestions.md](docs/OpenQuestions.md) | `OQ-##` unresolved questions, tracked not dropped | Harshdeep |

### Team and process

| Doc | Purpose | Owner |
|---|---|---|
| [Contributing.md](docs/Contributing.md) | Branch naming, commits, PR checklist, ownership map, dev loop | Anish |
| [Glossary.md](docs/Glossary.md) | Domain terms, acronyms, metric formulas, project jargon | Anish |

---

## Where things live — canonical ownership

Each fact has exactly one home. Link to it; never restate it.

| Content | Canonical location |
|---|---|
| Data model, field names, id scheme | [Schema.md](docs/Schema.md) |
| SQLite DDL for `structural.sqlite` | [Schema.md](docs/Schema.md) |
| Algorithm constants (RRF `k`, weights, thresholds) | [TechSpecifications.md](docs/TechSpecifications.md) |
| Environment variables | [Setup.md](docs/Setup.md) |
| Risk register `RISK-01`–`RISK-12` | [ImplementationPlan.md](docs/ImplementationPlan.md#5-risk-register) |
| Day plan and `M0`–`M7` milestone gates | [ImplementationPlan.md](docs/ImplementationPlan.md) |
| Task board `T-###`, burndown, eval run log | [Tracker.md](docs/Tracker.md) |
| Decisions and rejected alternatives | [Decisions.md](docs/Decisions.md) |
| Latency and index-build budgets | [TechSpecifications.md](docs/TechSpecifications.md), flows in [Appflow.md](docs/Appflow.md) |
| Locked facts: name, model stack, package layout, data model, budgets, targets | [_CONTRACT.md](docs/_CONTRACT.md) |

Docs that do **not** exist, and where that content lives instead:

| Expected name | Actual home |
|---|---|
| `Architecture.md` | [Design.md](docs/Design.md) |
| `Retrieval.md`, `Structural.md`, `Agent.md`, `Versioning.md` | [TechSpecifications.md](docs/TechSpecifications.md) — each is a section |
| `Evaluation.md` | [TestPlan.md](docs/TestPlan.md) for protocol; [Tracker.md](docs/Tracker.md) for the run log |
| `Risks.md` | [ImplementationPlan.md](docs/ImplementationPlan.md#5-risk-register) |
| `Roadmap.md` | [ImplementationPlan.md](docs/ImplementationPlan.md) for days; [Changelog.md](docs/Changelog.md) for releases |

---

## The one thing to understand before reading anything else

**The benchmark and the demo are different codebases, in different languages, and
this is deliberate.**

CoIR `AppsRetrieval` is built on APPS: English competitive-programming problem
statements retrieving **Python** solutions, each a standalone single file. The
JavaScript constraint in the problem statement applies to the **live demo
codebase**, not the screening benchmark.

Two consequences shape the whole system:

1. The structural AST/call-graph signal contributes ~nothing to NDCG@10, because
   APPS snippets have no cross-file call graph. It earns its place on the live demo,
   where P1 and Bonus are judged.
2. Sparse BM25 scores far below dense on APPS. The query is English prose, the document
   is Python source, and the two share almost no vocabulary — so equal-weight fusion with
   the sparse signal is a net negative, and `eval.yaml` down-weights it deliberately.
   The exact weight is `OQ-02`, swept on the **train** split only.

Axiom therefore ships two first-class profiles — `configs/eval.yaml` (dense-heavy,
structural off) and `configs/demo.yaml` (all three signals). See
[Decisions.md](docs/Decisions.md) for the ADRs and [OpenQuestions.md](docs/OpenQuestions.md#oq-01--is-the-two-profile-eval--demo-split-the-final-shape)
for the tracking entry.

---

## Conventions

- Status vocabulary: `Planned` → `In Progress` → `Blocked` → `Done` → `Dropped`.
- Identifier prefixes: `FR-##` functional requirement, `NFR-##` non-functional,
  `ADR-###` decision, `TC-###` test case, `RISK-##`, `OQ-##` open question,
  `NG-##` non-goal, `T-###` tracker task.
- Every doc opens with a purpose line, `**Owner:**`, `**Last updated:**`, `**Status:**`.
- Markdown only. No emoji. Tables where the content is tabular.
- Cross-reference by relative link. One canonical home per fact.

## Reference targets

The suite previously carried a `BGE 0.6B = 14.7` baseline that appears in neither of the papers it
was cited to; it has been removed rather than re-sourced. The headline claim is a **relative gain
over our own measured dense-only baseline `B`**, with an ablation table next to it. See
[_CONTRACT.md §8](docs/_CONTRACT.md#8-targets).

**`B` now exists.** Measured 2026-09-23 on the full CoIR `AppsRetrieval` test split (8,765 docs,
3,765 queries, no truncation), `all-MiniLM-L6-v2` INT8, rerank passthrough:

| Metric | Baseline `B` (measured) | Our target | Status |
|---|---|---|---|
| NDCG@10 (CoIR AppsRetrieval test) | **7.59** | ≥ 1.36 × `B` = **10.3** | not yet measured — no reranker weights exist |
| MRR@10 | **6.39** | ≥ 1.36 × `B_mrr` | not yet measured |
| Recall@100 (first stage) | **27.22** | — | **the binding constraint**: 73% of relevant docs never enter the candidate pool, and nothing downstream of retrieval can reach them |
| *(ablation)* sparse-only / hybrid RRF | 0.91 / 7.80 | — | hybrid is +0.21 NDCG (+2.76%) and **+0.00 recall** over `B` |
| Query p50 | — | budget: ≤ 900 ms ([_CONTRACT.md §7](docs/_CONTRACT.md#7-performance-budgets-locked-8-core-cpu--16-gb-ram-reference-box)) |
| Cold index, 10k chunks | — | budget: ≤ 12 min (same) |

A budget is not a measurement, and the two latency rows above are still budgets — **no latency or
memory figure in this project has ever been measured.** The accuracy rows *are* measured and are
logged in [Tracker.md §5](docs/Tracker.md#5-eval-metrics-log) and `artifacts/experiments.csv`, but
all three runs are stamped **`reportable: false`** (seven active placeholders, a dirty tree, and an
embedder that is not the configured primary). Nothing here may be quoted to the jury as a final
number until a reportable run replaces it.
