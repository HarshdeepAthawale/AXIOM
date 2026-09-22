# Contributing

Branch naming, commit conventions, the pre-review checklist, the ownership map, and the day-to-day dev loop for PRISM.

**Owner:** Anish Grover
**Last updated:** 2026-09-16
**Status:** Draft

Related: [Rules.md](Rules.md) · [Setup.md](Setup.md) · [ImplementationPlan.md](ImplementationPlan.md) · [Schema.md](Schema.md) · [Tracker.md](Tracker.md) · [Decisions.md](Decisions.md) · [Changelog.md](Changelog.md) · [Glossary.md](Glossary.md)

---

## How to read this document

[Rules.md](Rules.md) states the binding engineering invariants and the reviewer-side checklist that
enforces them. This document is the author-facing companion: how to structure your work so that
checklist passes on the first look, and the mechanical process — branches, commits, PRs, recipes —
that a four-person team needs written down once rather than re-negotiated every day of a 10-day
sprint. Nothing here overrides [Rules.md](Rules.md); where the two could be read as disagreeing,
[Rules.md](Rules.md) wins.

---

## 1. Branch naming

One scheme, four people, no ambiguity:

```
<initials>/<area>-<short-description>
```

| Field | Values |
|---|---|
| `initials` | `prab` (Prabinder), `anish` (Anish), `harsh` (Harshdeep), `parth` (Parth) |
| `area` | `retrieval`, `structural`, `agent`, `rerank`, `versioning`, `schema`, `api`, `ui`, `eval`, `docs` — matches the workstream areas in [ImplementationPlan.md §2](ImplementationPlan.md#2-workstreams) |
| `short-description` | 2–4 words, kebab-case, what the branch does, not the ticket number (there are no ticket numbers — `T-###` ids live in [Tracker.md](Tracker.md) and belong in the commit/PR body, not the branch name) |

Examples: `prab/retrieval-dense-index`, `anish/structural-chunker-fallback`,
`harsh/agent-sufficiency-predicate`, `parth/versioning-incremental-reindex`,
`anish/docs-contributing-glossary`.

`schema/` changes get their own prefix regardless of who authors them —
`schema/<initials>-<short-description>` — so a schema PR is visually distinct in the branch list
before anyone opens it, since it always needs the four-member sign-off in
[§3](#3-pr-checklist) below.

No `feature/`, `bugfix/`, `hotfix/` prefixes — the area already carries that information, and a
10-day sprint with four people does not need a taxonomy built for a larger org.

---

## 2. Commit conventions

A lightweight conventional-commit prefix, one line per commit, imperative mood:

```
<type>(<area>): <summary>
```

| Type | Use for |
|---|---|
| `feat` | New capability — a new `FR-##` landing, a new CLI subcommand, a new endpoint |
| `fix` | Bug fix, no contract change |
| `perf` | Performance work with no behaviour change — **the PR body must include before/after `bench_latency.py` output**, per [Rules.md §9.6](Rules.md#96-performance-regression-policy) item 2 |
| `refactor` | Internal restructuring, no observable behaviour change |
| `test` | Test-only change |
| `docs` | `docs/`-only change |
| `chore` | Tooling, CI, dependency bump — **a dependency addition needs a PR-body line stating what it replaces or enables**, per [Rules.md §9.4](Rules.md#94-dependency-pinning) item 3 |

`area` matches the branch-name area vocabulary in [§1](#1-branch-naming). Examples:

```
feat(retrieval): implement weighted RRF fusion with per-QueryType weights
fix(structural): guard against circular imports during graph traversal
perf(agent): cut classifier heuristic regex compilation to import time
docs(contributing): add Recipe B for new config flags
chore(schema): pin faiss-cpu==1.8.0
```

A commit that touches `src/axiom/schema/` states so in its body even when the subject line's area
is something else, because schema diffs are searched for specifically during review
([Rules.md §10](Rules.md#10-code-review-rules) item 7).

---

## 3. PR checklist

[Rules.md §10](Rules.md#10-code-review-rules) is the **authoritative reviewer-side checklist** —
read it, it is not reproduced here. This section is the same checklist run by the **author**,
before requesting review, so that a reviewer's first pass finds nothing that a five-minute
self-check would have caught. Run through [Rules.md §10](Rules.md#10-code-review-rules) items 1–10
yourself, then answer these two PRISM-specific questions that the reviewer-side list assumes you
already asked:

| Question | If yes |
|---|---|
| Is this PR `docs/`-only? | Self-merge is allowed per [Rules.md §10](Rules.md#10-code-review-rules) etiquette — no review wait required, but still open a normal PR so the change is visible in history. |
| Does this PR touch `src/axiom/schema/`? | It needs **four-member sign-off**, not a single reviewer's approval — see [Rules.md §9.3](Rules.md#93-type-hints) item 5 ("Do not pass `dict[str, Any]` between stages — if a stage needs a new field, add it to Schema.md and the model, with the four-member sign-off that schema changes require"). This exists because a schema change is the one class of PR that can break all four workstreams simultaneously — see [ImplementationPlan.md `RISK-12`](ImplementationPlan.md#risk-12--a-late-schema-change-breaks-multiple-workstreams-at-once). Get the sign-off *before* merging, recorded as four approving reviews or four explicit comments on the PR, not after. |

The full **"Blocks merge (no discussion, fix it)"** list — mutated `chunk_id`, hardcoded tunable,
`except: pass`, `print` in `src/axiom/`, a CUDA reference, a failing test, a `mypy --strict` error in
a strict package, a committed data/weight/index artefact, an unremoved `# PLACEHOLDER` in a reported
score, a schema change without sign-off, a latency regression > 10% without an ADR — lives in
[Rules.md §10](Rules.md#10-code-review-rules) and is not restated here; treat it as this checklist's
final gate by reference.

---

## 4. Ownership map

Summarised from [ImplementationPlan.md §2](ImplementationPlan.md#2-workstreams) — that table is
authoritative; this is the quick-reference view plus the review pairing this document owns.

| Workstream | Owner | Area |
|---|---|---|
| Retrieval core | Prabinder Singh | `indexing/dense.py`, `indexing/sparse.py`, `retrieval/dense.py`, `retrieval/sparse.py`, `retrieval/fusion.py`, `eval/` |
| Structural intelligence | Anish Grover | `chunking/`, `indexing/structural.py`, `retrieval/structural.py` |
| Agentic orchestration + reranking + UI | Harshdeep Athawale | `agent/`, `rerank/`, `api/`, `ui/` |
| Versioning + evaluation + submission | Parth Deshmukh | `versioning/`, release packaging |
| Cross-cutting (all four) | — | `schema/`, `core/`, `configs/`, `docs/` |

### 4.1 Cross-review pairing

Nobody reviews only their own workstream's PRs — a four-person team where each owner is also the
sole reviewer of their own area defeats the point of review. Pairing follows the actual data
dependency between workstreams, so each reviewer already understands the boundary they're checking:

| Author's workstream | Primary cross-reviewer | Why this pairing |
|---|---|---|
| Retrieval core (Prabinder) | Harshdeep | Fusion's output (`FusedResult`) is exactly what the reranker consumes — the retrieval/agent boundary is the one place a silent contract drift between the two workstreams would be invisible to either owner alone. |
| Agentic orchestration + rerank (Harshdeep) | Prabinder | Same boundary, reviewed from the other side. |
| Structural intelligence (Anish) | Parth | The chunker's `ChunkMetadata` (`version_id`, `commit_sha`, `last_modified`) is exactly what the versioning workstream depends on for provenance — a chunker change that drops or mis-populates a provenance field breaks versioning silently, not loudly. |
| Versioning (Parth) | Anish | Same boundary, reviewed from the other side. |

Either pairing member may also review anything else — this table sets the *default* reviewer, not
an exclusive one, and a same-day availability gap should never block a PR past the 4-hour etiquette
window in [Rules.md §10](Rules.md#10-code-review-rules).

---

## 5. Dev loop

The day-to-day cycle, every day of the sprint:

1. **Pull** `main`.
2. **Branch** per [§1](#1-branch-naming).
3. **Sync dependencies**: `uv sync --frozen` (see [Setup.md §4.1](Setup.md#41-primary-path--uv)). If
   `--frozen` fails, `uv.lock` and `pyproject.toml` have drifted on `main` — fix that first, in its
   own PR, before starting feature work on top of a broken lock.
4. **Make the change**, following [Rules.md](Rules.md) as you go rather than as a post-hoc check —
   the anti-pattern gallery in [Rules.md §11](Rules.md#11-anti-pattern-gallery) is written to be read
   *before* writing the code it warns about, not after a review comment.
5. **Run the verification ladder locally before pushing.** [Setup.md §8](Setup.md#8-verification-ladder)'s
   five rungs (imports/native libs, package wiring, unit tests, smoke index, smoke search) are
   written as a fresh-clone setup check, but rungs 3–5 double as exactly the right local pre-push
   check for a change to any of `chunking/`, `indexing/`, `retrieval/`, `rerank/`, or `agent/` — run
   `pytest -q tests/ -x --timeout=120` at minimum, and the smoke index/search rungs for anything that
   touches the on-disk index format.
6. **Open the PR**, running [§3](#3-pr-checklist) above yourself first.
7. **Address review within 4 hours** during the sprint, per
   [Rules.md §10](Rules.md#10-code-review-rules) etiquette — say `blocking:` or `nit:` explicitly on
   every comment you leave as a reviewer, and resolve as the author; the reviewer closes.
8. **Merge.** Squash or merge-commit, either is fine for this project size; what matters is that
   `main` always passes CI stages 1–5 ([TestPlan.md §8](TestPlan.md#8-ci-pipeline)).

---

## Recipe A — add a new `FR-##` requirement

A new functional requirement is scope growth, so it does not enter quietly:

1. Propose it to the [PRD.md](PRD.md) owner (Parth) with the user story it serves and which persona
   ([PRD.md §3](PRD.md#3-personas)) it's for.
2. Get explicit sign-off — for a hackathon-scale team this can be a same-day Slack/standup
   confirmation, but it must be confirmed by more than the proposer alone, since every `FR-##`
   competes for the same 10-day window as everything already committed in
   [PRD.md §7](PRD.md#7-prioritisation-moscow).
3. Add the `FR-##` to [PRD.md §5](PRD.md#5-functional-requirements) with an owner and a priority
   (`P0`/`P1`/`B`/`INF`), and slot it into the MoSCoW table and, if it's a Must-have, the cut-order
   list in [PRD.md §7.1](PRD.md#71-cut-order-under-time-pressure).
4. Add the corresponding `TC-###` acceptance test(s) to [TestPlan.md](TestPlan.md) in the right
   category — a requirement with no test case is not yet real.
5. Add the corresponding `T-###` task(s) to [Tracker.md](Tracker.md), in the block owned by the
   requirement's `FR-##` owner, with a target day consistent with
   [ImplementationPlan.md §4](ImplementationPlan.md#4-day-by-day-plan)'s existing schedule — do not
   silently assume a new requirement is free; state which day it lands and what, if anything, it
   displaces.

---

## Recipe B — add a new config flag

Every tunable lives in `src/axiom/config.py` or a `configs/*.yaml` profile — never at a callsite,
per [Rules.md §7](Rules.md#7-configuration-discipline). Adding one:

1. **Add the `Settings` field** in `src/axiom/config.py` with a documented one-line description and
   an explicit unit in the field name (`_s`, `_ms`, `_mb` — not bare `timeout`/`budget`/`size`, per
   [Rules.md §7](Rules.md#7-configuration-discipline)'s docstring-per-field rule). If the value is a
   design estimate rather than a measured one, mark it with a `# PLACEHOLDER` comment and a tracker
   id, per the exact convention in [Rules.md §8](Rules.md#8-the-placeholder-convention).
2. **Add it to every `configs/*.yaml` profile it applies to** (`default`, `fast`, `accurate`,
   `eval`, `demo` — not necessarily all five; state in the PR body which profiles the flag is
   relevant to and why the others are unaffected).
3. **Add it to [Setup.md §7](Setup.md#7-environment-variables)'s environment-variable table** — that
   table is the authoritative reference for every `AXIOM_`-prefixed variable; do not leave a new
   flag undocumented there even if it's also described inline in `config.py`.
4. **Write a test** that exercises the flag's effect — a config field with no test asserting its
   behaviour is indistinguishable from a field that does nothing.
5. **Note it in [Changelog.md](Changelog.md)** if it changes default retrieval behaviour (a new flag
   defaulting to a value that changes what a query returns is a behaviour change even though it is
   not a schema change — see [Changelog.md](Changelog.md#format) for the breaking/non-breaking
   distinction that applies here).

This is the exact procedure [Rules.md §7](Rules.md#7-configuration-discipline)'s table points to as
"[Contributing.md](Contributing.md#recipe-b--add-a-new-config-flag)" — if you arrived here from that
link, you're in the right place.

---

## Related documents

| Document | Relationship |
|---|---|
| [Rules.md](Rules.md) | The binding invariants this document's checklist operationalises |
| [Setup.md](Setup.md) | The verification ladder run locally in the dev loop |
| [ImplementationPlan.md](ImplementationPlan.md) | The workstream table this document's ownership map summarises |
| [Tracker.md](Tracker.md) | Where `T-###` tasks from Recipe A land |
| [Glossary.md](Glossary.md) | Definitions for the jargon used above |
