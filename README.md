# PRISM Documentation

Index for the **PRISM — Agentic Code Intelligence** documentation suite.

**Owner:** Harshdeep Athawale
**Last updated:** 2026-09-15
**Status:** Draft

---

## What PRISM is

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
| Build window | 2026-09-15 → 2026-09-27 |
| Release tag | `PRISM_GENAI_HACKATHON_Y2026` |

---

## Read in this order

New to the project? Follow this path.

1. [PRD.md](PRD.md) — what we are building and why; requirements and success metrics
2. [Design.md](Design.md) — architecture, diagrams, and the reasoning behind the shape
3. [Schema.md](Schema.md) — the shared data model; **all four workstreams code against this**
4. [TechSpecifications.md](TechSpecifications.md) — component-by-component engineering spec
5. [Setup.md](Setup.md) — get it running from a clean clone
6. [Rules.md](Rules.md) — the binding engineering invariants; read before your first commit

---

## Full index

### Core

| Doc | Purpose | Owner |
|---|---|---|
| [PRD.md](PRD.md) | Problem, personas, `FR-##` / `NFR-##` requirements, success metrics, jury alignment | Parth |
| [TechSpecifications.md](TechSpecifications.md) | Runtime, model stack, per-module specs, algorithms, config reference | Prabinder |
| [Appflow.md](Appflow.md) | Eight end-to-end runtime flows with sequence diagrams and budgets | Harshdeep |
| [Design.md](Design.md) | Architecture, principles, concurrency model, degradation ladder, extension points | Anish |
| [Schema.md](Schema.md) | Pydantic data model, id scheme, on-disk formats, SQLite DDL, invariants | Prabinder |
| [ImplementationPlan.md](ImplementationPlan.md) | Workstreams, day-by-day plan, `M0`–`M7` gates, `RISK-01`–`RISK-12` | Prabinder |
| [Tracker.md](Tracker.md) | `T-###` task board, burndown, standup log, eval metrics log | Parth |
| [Rules.md](Rules.md) | Cardinal rules, engineering invariants, anti-pattern gallery | Harshdeep |

### Decisions and history

| Doc | Purpose | Owner |
|---|---|---|
| [Decisions.md](Decisions.md) | `ADR-###` log: what was chosen, why, alternatives rejected | Prabinder |
| [Changelog.md](Changelog.md) | What shipped and when; breaking vs non-breaking; planned releases | Parth |

### Setup and ops

| Doc | Purpose | Owner |
|---|---|---|
| [Setup.md](Setup.md) | Prerequisites, install paths, env vars, verification ladder, troubleshooting | Parth |
| [Deployment.md](Deployment.md) | Docker, submission runbook, rollback, demo-day runbook | Parth |

### Quality and safety

| Doc | Purpose | Owner |
|---|---|---|
| [TestPlan.md](TestPlan.md) | `TC-###` cases, edge catalogue, performance tests, eval protocol, CI | Parth |
| [Security.md](Security.md) | Threat model (STRIDE), trust boundaries, data handling, dependency security | Harshdeep |

### API and integration

| Doc | Purpose | Owner |
|---|---|---|
| [API.md](API.md) | HTTP endpoint contracts, error codes, CLI reference | Harshdeep |

### Scope control

| Doc | Purpose | Owner |
|---|---|---|
| [NonGoals.md](NonGoals.md) | `NG-##` explicitly out-of-scope items and what we do instead | Parth |
| [OpenQuestions.md](OpenQuestions.md) | `OQ-##` unresolved questions, tracked not dropped | Harshdeep |

### Team and process

| Doc | Purpose | Owner |
|---|---|---|
| [Contributing.md](Contributing.md) | Branch naming, commits, PR checklist, ownership map, dev loop | Anish |
| [Glossary.md](Glossary.md) | Domain terms, acronyms, metric formulas, project jargon | Anish |

---

## Where things live — canonical ownership

Each fact has exactly one home. Link to it; never restate it.

| Content | Canonical location |
|---|---|
| Data model, field names, id scheme | [Schema.md](Schema.md) |
| SQLite DDL for `structural.sqlite` | [Schema.md](Schema.md) |
| Algorithm constants (RRF `k`, weights, thresholds) | [TechSpecifications.md](TechSpecifications.md) |
| Environment variables | [Setup.md](Setup.md) |
| Risk register `RISK-01`–`RISK-12` | [ImplementationPlan.md](ImplementationPlan.md#risk-register) |
| Day plan and `M0`–`M7` milestone gates | [ImplementationPlan.md](ImplementationPlan.md) |
| Task board `T-###`, burndown, eval run log | [Tracker.md](Tracker.md) |
| Decisions and rejected alternatives | [Decisions.md](Decisions.md) |
| Latency and index-build budgets | [TechSpecifications.md](TechSpecifications.md), flows in [Appflow.md](Appflow.md) |

Docs that do **not** exist, and where that content lives instead:

| Expected name | Actual home |
|---|---|
| `Architecture.md` | [Design.md](Design.md) |
| `Retrieval.md`, `Structural.md`, `Agent.md`, `Versioning.md` | [TechSpecifications.md](TechSpecifications.md) — each is a section |
| `Evaluation.md` | [TestPlan.md](TestPlan.md) for protocol; [Tracker.md](Tracker.md) for the run log |
| `Risks.md` | [ImplementationPlan.md](ImplementationPlan.md#risk-register) |
| `Roadmap.md` | [ImplementationPlan.md](ImplementationPlan.md) for days; [Changelog.md](Changelog.md) for releases |

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
2. BM25 scores **4.8** NDCG@10 on APPS against BGE-0.6B's **14.7**. The query is prose,
   the document is code, and they share almost no vocabulary. Equal-weight fusion with
   a 4.8-scoring signal is a net negative.

PRISM therefore ships two first-class profiles — `configs/eval.yaml` (dense-heavy,
structural off) and `configs/demo.yaml` (all three signals). See
[Decisions.md](Decisions.md) for the ADRs and [OpenQuestions.md](OpenQuestions.md#oq-01)
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

| Metric | Reference | Our target |
|---|---|---|
| NDCG@10 (CoIR AppsRetrieval test) | BM25 4.8 · BGE-0.6B 14.7 · E5-Mistral-7B 23.5 · SOTA 26.5 | ≥ 20.0 |
| MRR | — | ≥ 22.0 |
| Recall@100 (first stage) | — | ≥ 65.0 |
| Query p50 | — | ≤ 900 ms |
| Cold index, 10k chunks | — | ≤ 12 min |
