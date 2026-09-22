# Deployment

Docker build, the submission runbook for cutting `PRISM_GENAI_HACKATHON_Y2026`, rollback posture, and the demo-day runbook.

**Owner:** Parth Deshmukh
**Last updated:** 2026-09-16
**Status:** Draft

Related: [Setup.md](Setup.md) · [PRD.md](PRD.md) · [ImplementationPlan.md](ImplementationPlan.md) · [Tracker.md](Tracker.md) · [TestPlan.md](TestPlan.md) · [NonGoals.md](NonGoals.md) · [Security.md](Security.md) · [Decisions.md](Decisions.md) · [_CONTRACT.md](_CONTRACT.md)

---

## 1. Scope

This document covers everything that happens **after** [Setup.md](Setup.md) gets a clean clone
running locally: packaging that same system into a container, cutting the submission release, and
the exact sequence executed on demo day. It does not repeat prerequisites, environment variables, or
the verification ladder — those are [Setup.md](Setup.md)'s subject.

Deployment here means one thing only: **a container the evaluator runs on their own machine.** Per
[`NG-13`](NonGoals.md#ng-13--no-hosted-cloud-deployment), nothing in this document stands up a
service anyone but the evaluator controls. There is no URL. There is no billing account whose
uptime we depend on. "Deployment" is Docker plus a runbook, not infrastructure.

---

## 2. What ships in the image, and what does not

| Ships in the image | Pulled at first run, into a mounted volume |
|---|---|
| `src/axiom/`, `configs/*.yaml`, `pyproject.toml`, `uv.lock` | Model weights (embedder, reranker, query LLM GGUF) — see [Setup.md §5](Setup.md#5-models-and-dataset) |
| The Python 3.11 interpreter and pinned dependency set | The CoIR `AppsRetrieval` dataset cache |
| `tests/fixtures/mini_repo/` (the 14-file smoke fixture — tiny, deterministic, exists precisely so the image can self-verify) | Any target repository being indexed (`AXIOM_INDEX_ROOT`, the user's own corpus) |
| Nothing under `.axiom/` — index artefacts are always derived, per [Rules.md §9.5](Rules.md#95-what-may-and-may-not-be-committed) | The `.axiom/` index tree itself |

Baking multi-gigabyte model weights into the image would violate [`NG-13`](NonGoals.md#ng-13--no-hosted-cloud-deployment)'s
spirit even though it does not technically host anything — it turns the image build into a slow,
network-dependent step that has to happen on *our* machine before we can ship, instead of on the
evaluator's, where `HF_HOME` caching and the pre-download step already handle it once and for all
([Setup.md §5.4](Setup.md#54-pre-download-for-an-offline-demo-do-this-the-day-before)). The image
stays small, builds fast, and the weights live in a bind-mounted `data/` directory that survives a
container rebuild.

`NFR-12`'s budget governs the mounted volume, not the image: on-disk index for 10k chunks stays
under 1.5 GB including shared blobs; the primary model profile's total download stays under 2.5 GB;
the fallback profile stays under 500 MB, so a bandwidth-limited evaluator running `configs/fast.yaml`
can still reach a working demo.

---

## 3. Docker

### 3.1 Dockerfile

Single-stage, CPU-only, `python:3.11-slim-bookworm` — the exact base named in `_CONTRACT.md §1` and
the interpreter version locked by [`NG-21`](NonGoals.md#ng-21--no-python-313-support). The CPU-torch
install ordering from [Setup.md §3](Setup.md#3-install-cpu-torch-first-mandatory-all-platforms) is
**mandatory** here too — reversing it inside the container produces the same ~2.5 GB of dead
`nvidia-*` wheels that §3 exists to prevent, except now baked irreversibly into an image layer.

```dockerfile
# Dockerfile
FROM python:3.11-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_VISIBLE_DEVICES="" \
    HF_HOME=/app/data/hf \
    AXIOM_INDEX_ROOT=/app/.axiom \
    AXIOM_DATA_ROOT=/app/data

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

WORKDIR /app

# --- CPU torch FIRST, exactly as Setup.md §3 mandates. Order matters: a later
# `uv sync` must never be allowed to re-resolve torch into a CUDA build. ---
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY src/ src/
COPY configs/ configs/
COPY tests/fixtures/mini_repo/ tests/fixtures/mini_repo/
RUN uv sync --frozen

# Non-root: nothing in this project needs root, and the mounted data/ and
# .axiom/ volumes should not end up owned by root on the host.
RUN useradd -m -u 1000 axiom && chown -R axiom:axiom /app
USER axiom

ENTRYPOINT ["uv", "run", "axiom"]
CMD ["--help"]
```

Verify the CPU-only build the same way [Setup.md §3](Setup.md#3-install-cpu-torch-first-mandatory-all-platforms)
verifies a local install, just inside the container:

```bash
docker build -t axiom-retrieval:local .
docker run --rm axiom-retrieval:local -c \
  "python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\""
# expected: 2.4.1+cpu False
```

`CUDA_VISIBLE_DEVICES=""` is set in the image itself, not only in CI, matching
[Rules.md §6](Rules.md#6-cpu-only-enforcement) item 2's requirement that it be set "in CI, in the
Dockerfile, and in `scripts/run_eval.py`."

### 3.2 `docker-compose.yml`

Two services — the API and the Streamlit UI — sharing one model cache and one index volume, on one
Docker network so the UI can reach the API by service name.

```yaml
# docker-compose.yml
services:
  api:
    build: .
    entrypoint: ["uv", "run", "axiom"]
    command: ["serve", "--host", "0.0.0.0", "--port", "8000"]
    environment:
      AXIOM_API_HOST: "0.0.0.0"
      AXIOM_API_PORT: "8000"
      AXIOM_PROFILE: "${AXIOM_PROFILE:-default}"
      AXIOM_OFFLINE: "${AXIOM_OFFLINE:-false}"
    volumes:
      - ./data:/app/data
      - ./.axiom:/app/.axiom
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/v1/health"]
      interval: 10s
      timeout: 5s
      retries: 6

  ui:
    build: .
    entrypoint: ["uv", "run", "streamlit"]
    command: ["run", "src/axiom/ui/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
    environment:
      AXIOM_UI_PORT: "8501"
      # The one line that matters: inside the compose network the UI reaches
      # the API by SERVICE NAME, never by 127.0.0.1/localhost. Each service
      # is its own network namespace; 127.0.0.1 inside the `ui` container
      # means the `ui` container, which is not running the API.
      AXIOM_API_BASE_URL: "http://api:8000"
    volumes:
      - ./data:/app/data
      - ./.axiom:/app/.axiom
    ports:
      - "8501:8501"
    depends_on:
      api:
        condition: service_healthy
```

This is the exact gotcha [Setup.md §9](Setup.md#9-troubleshooting)'s troubleshooting table
forward-references: *"Streamlit UI loads but every query shows 'API unreachable' ... Inside Docker
the UI must use the compose service name, not `127.0.0.1` — see Deployment.md §3."* The fix is the
`AXIOM_API_BASE_URL: "http://api:8000"` line above — `api` is the compose service name, resolved by
Docker's embedded DNS, and it is the only value that works from inside the `ui` container. Running
the same two processes outside Docker (bare `axiom serve` + `streamlit run`) correctly uses
`http://127.0.0.1:8000` instead, per [Setup.md §7.6](Setup.md#76-services) — the value is
environment-specific by design, not a bug in either environment.

Bring the stack up:

```bash
docker compose up --build
# once healthy:
curl -s http://localhost:8000/v1/health
open http://localhost:8501   # or: xdg-open / start, depending on platform
```

### 3.3 Image size and build-time budget

| Layer | Approx size | Notes |
|---|---|---|
| `python:3.11-slim-bookworm` base | ~120 MB | |
| `apt-get` toolchain (`build-essential`, `git`, `curl`) | ~200 MB | needed only if a wheel falls back to a source build; kept because `tree-sitter-javascript` wheels are not guaranteed on every platform this image might be rebuilt for |
| CPU torch + `uv sync` dependency set | ~900 MB | dominated by `onnxruntime`, `faiss-cpu`, `llama-cpp-python`, CPU torch |
| Application code (`src/`, `configs/`, fixtures) | < 5 MB | |
| **Total image** | **~1.2 GB** | model weights excluded — see §2 |

Budget: image build completes in **under 4 minutes** on the reference box with a warm `pip`/`uv`
cache, under 8 minutes cold. This is a build-time budget, not one of the `_CONTRACT.md §7` runtime
performance budgets — it exists so a same-day image rebuild (e.g. after a Day-9 bug fix, per
[ImplementationPlan.md §4](ImplementationPlan.md#4-day-by-day-plan)) never becomes the bottleneck on
a day that is already tight.

---

## 4. Submission runbook

The sequence that turns a green `M7` gate ([ImplementationPlan.md §3](ImplementationPlan.md#3-milestone-gates-m0m7))
into the actual submitted artefact. Executed once, on Day 10, by Parth, with all four members
available to unblock anything that surfaces.

### 4.1 Preconditions

All seven items of [PRD.md §2.1 Definition of Done](PRD.md#21-definition-of-done) are
**simultaneously** true — not sequentially achieved and possibly regressed by the time the last one
lands. Re-verify all seven the same day the tag is cut, not on whichever earlier day each first went
green:

1. `axiom index <repo>` completes inside the cold-index budget on the demo repo.
2. `axiom query "<Q1|Q2|Q3>"` returns file+line results for all three archetypes.
3. `scripts/run_eval.py` produces `appsretrieval_results.json` with NDCG@10 ≥ 20.0.
4. `axiom reindex --to <newer-commit>` completes inside 45 s for a 50-file diff.
5. `axiom query --all-versions "<query>"` returns `SnippetFamily`-collapsed results.
6. The whole pipeline runs green with `AXIOM_LLM_ENABLED=false`.
7. Release `PRISM_GENAI_HACKATHON_Y2026` exists with the eval JSON attached — this section produces
   that seventh item.

Corresponding tasks in [Tracker.md §2.6](Tracker.md#26-evaluation-discipline-submission-docs-t-191t-230):
`T-220` (PPT), `T-221` (release cut), `T-222` (final Definition-of-Done check), `T-230` (form
submission).

### 4.2 Sequence

```bash
# 1. Final reportable eval run — the ONLY run whose number is ever quoted,
#    per TestPlan.md §6.2. Must be a clean tree, not `-dirty`.
git status --porcelain   # must be empty
python scripts/run_eval.py --task AppsRetrieval --split test --profile eval \
  --out appsretrieval_results.json

# 2. Confirm no # PLACEHOLDER constant reached this run unresolved
#    (Rules.md §8 item 3 — the run banner lists any that remain).
grep -c PLACEHOLDER appsretrieval_results.json || true   # expect: no match

# 3. Log the run in the experiment log (TestPlan.md §6.4 / Tracker.md §5)
#    BEFORE tagging, so the tag's commit and the logged row agree.
#    Append the row to data/experiments.csv, commit it.
git add data/experiments.csv appsretrieval_results.json
git commit -m "eval: final reportable run for submission"

# 4. Tag. The tag name is organiser-mandated and does not change even though
#    the project itself is named Axiom now (ADR-015) — see OQ-04.
git tag -a PRISM_GENAI_HACKATHON_Y2026 -m "Samsung PRISM GenAI Hackathon 3rd Edition submission"
git push origin PRISM_GENAI_HACKATHON_Y2026

# 5. Cut the GitHub Release from that tag, attach the eval JSON as a release
#    artifact (not just committed in-repo — the submission checklist in
#    PROJECT_OVERVIEW.md §11 requires it attached to the Release itself).
gh release create PRISM_GENAI_HACKATHON_Y2026 \
  appsretrieval_results.json \
  --title "Axiom — PRISM GenAI Hackathon 3rd Edition Submission" \
  --notes-file RELEASE_NOTES.md
```

### 4.3 What the release must contain

Per `PROJECT_OVERVIEW.md §11` and [PRD.md §2.1](PRD.md#21-definition-of-done):

| Artefact | Where |
|---|---|
| Public GitHub repo at the tagged commit | the tag itself |
| `README.md` with reproducible setup (5-command path + Docker) | repo root, `T-210` |
| `appsretrieval_results.json` | attached to the GitHub Release |
| Demo video (≤ 5 min) | linked from the Release notes and the Google Form |
| `Incognito_Submission_ppt` | linked from the Release notes and the Google Form |
| Google Form submission with all links | `T-230`, separate from the git-side steps above |

### 4.4 Post-tag change policy

[Rules.md §12](Rules.md#12-rule-change-procedure) item 4 already states rules may not be relaxed in
the final two days; the same discipline applies to the tag. Once
`PRISM_GENAI_HACKATHON_Y2026` is pushed, no further commit changes what the tag points at — see
§5 for what happens if a defect is found afterward.

---

## 5. Rollback

There is no production system here to roll back — [`NG-10`](NonGoals.md#ng-10--no-production-sla-uptime-or-ha-guarantee)
is explicit that Axiom carries no SLA, uptime, or HA guarantee, and inventing a rollback apparatus
for a system with no live deployment (`NG-13`) would be solving a problem this project does not
have. "Rollback" here means exactly one thing: **what to do if a defect in the tagged submission is
discovered after `PRISM_GENAI_HACKATHON_Y2026` is pushed.**

| Situation | Correct action |
|---|---|
| The eval JSON is found to be stale (built from an older commit than the tag) | Re-run §4.2 steps 1–3 against the current tagged commit, re-attach the corrected `appsretrieval_results.json` to the *same* Release via `gh release upload --clobber`. The tag itself does not move — only the Release artifact is corrected. |
| A P0 defect is found in the tagged commit before the 27 Sep 11:59 PM deadline | Fix on a branch, merge to `main`, **create a new, later commit**, and **retag**: `git tag -a PRISM_GENAI_HACKATHON_Y2026 -f -m "..."` then `git push -f origin PRISM_GENAI_HACKATHON_Y2026`. This is one of the few sanctioned uses of a force-push in this project, and only to a tag, never to `main` itself, and only before the deadline. |
| A P0 defect is found after the deadline | Per the hackathon's own buffer-day allowance (`PROJECT_OVERVIEW.md §13`, "Buffer — Deadline day"), the same retag procedure applies up to 27 Sep 11:59 PM; after that, the submission is what it is — there is no post-deadline rollback path, because the organisers' own submission window is the actual boundary, not something this project's tooling can extend. |
| The demo repo (`OQ-07`) turns out to have a licensing or provenance problem discovered late | Swap it and rebuild the P1/Bonus artefacts before retagging; never ship a corpus whose provenance is not clean, regardless of schedule pressure. |

No blue/green, no canary, no staged rollout — a single retag is the entire mechanism, and it is
used rarely and deliberately, not as a routine part of the release process.

---

## 6. Environment parity

The judge's machine is unknown hardware, running our container or our bare-metal setup, once,
without our help. Parity with it — not with our own laptops — is the thing that actually matters.

| | Our dev machines | CI (`.github/workflows/ci.yml`) | Judge's machine (assumed) |
|---|---|---|---|
| OS | Linux, macOS arm64, Windows x86_64 ([Setup.md §2](Setup.md#2-platform-notes)) | Ubuntu (GitHub-hosted runner) | Unknown — Linux, Windows, or macOS |
| Python | 3.11 or 3.12 | 3.11 and 3.12 matrix | Whatever the container provides: **3.11**, pinned ([`NG-21`](NonGoals.md#ng-21--no-python-313-support)) |
| GPU | None used regardless of what's physically present | None (hosted runner) | Assumed none; CPU-only is the entire point of `NFR-06` |
| Network | Available for setup, `AXIOM_OFFLINE` rehearsed for demo | Available; `pip-audit` and model-cache steps depend on it | **Assumed unreliable** — venue wifi is the risk `NG-13` and §7 below are built around |
| Architecture | x86_64 and arm64 (macOS) | x86_64 | Assumed x86_64; Windows-arm64 has no supported native path (`NG-28`) — the container path in §3 sidesteps this entirely, since Docker Desktop on Windows-arm64 still runs an x86_64 Linux guest |

The container is the parity mechanism: running `docker compose up` gives the judge the exact
`python:3.11-slim-bookworm` + pinned-dependency environment CI already validates, regardless of what
their host OS is. The bare-metal path in [Setup.md](Setup.md) is the fallback for a judge who
prefers not to use Docker, and it is why [Setup.md §2](Setup.md#2-platform-notes)'s per-platform
notes exist at all — Docker collapses that whole table to one row.

---

## 7. Demo-day runbook

This is the runbook [Setup.md §9](Setup.md#9-troubleshooting)'s troubleshooting table
forward-references: *"Never let the jury trigger the cold path — see the demo runbook in
Deployment.md §7."* Executed by the presenter (Harshdeep, per the demo-video/UI ownership in
[ImplementationPlan.md §2](ImplementationPlan.md#2-workstreams)), twice: once before recording the
demo video, once again immediately before the live jury session, per
[TestPlan.md §7](TestPlan.md#7-manual-test-script-demo-day)'s own instruction that the manual test
script is "Executed by the presenter before recording and again before the live session."

### 7.1 The day before (T-minus-1)

1. Pre-download every model — primary **and** fallback — per
   [Setup.md §5.4](Setup.md#54-pre-download-for-an-offline-demo-do-this-the-day-before):
   ```bash
   export HF_HOME="$PWD/data/hf"
   huggingface-cli download Qwen/Qwen3-Embedding-0.6B
   huggingface-cli download BAAI/bge-reranker-v2-m3
   huggingface-cli download sentence-transformers/all-MiniLM-L6-v2
   huggingface-cli download cross-encoder/ms-marco-MiniLM-L-6-v2
   huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF \
     --include "qwen2.5-1.5b-instruct-q4_k_m.gguf" \
     --local-dir data/models/gguf --local-dir-use-symlinks False
   python -c "import mteb; mteb.get_task('AppsRetrieval').load_data()"
   ```
2. Flip hard-offline and prove the system still works end to end:
   ```bash
   export HF_HUB_OFFLINE=1
   export AXIOM_OFFLINE=true
   axiom search "how is the input normalized" --top-k 5
   ```
3. Run the manual test script in full —
   [TestPlan.md §7](TestPlan.md#7-manual-test-script-demo-day), cases `M-01` through `M-13` — and
   record every result in [Tracker.md §6](Tracker.md#6-demo-day-readiness-checklist). Any FAIL on
   `M-01`..`M-03` or `M-08`..`M-10` blocks recording, per that section's own rule; fix and re-run the
   full script, not just the failed case, since an unrelated fix can regress an earlier pass.
4. Rebuild and smoke-test the Docker path too if the demo will run from the container rather than
   bare metal — `docker compose up --build`, then re-run the health check and one query from §3.2.

### 7.2 Immediately before recording

1. Re-run the manual test script (§7.1 step 3) one more time on the exact machine and exact index
   that will be used for the recording — a passing run the day before does not guarantee the
   recording machine's state is identical.
2. **Warm the process.** Cold model load (ONNX session construction, GGUF mmap) takes 20–40 s on
   first query, per [Setup.md §9](Setup.md#9-troubleshooting)'s troubleshooting table — this is
   expected, not a bug, but it must never be what the recording shows:
   ```bash
   curl -s "http://localhost:8000/v1/health?warm=true"
   ```
   Confirm the response reports all models loaded before starting to record.
3. Record. Show the wall-clock timer on screen for `M-13` per
   [TestPlan.md §7](TestPlan.md#7-manual-test-script-demo-day), and show `M-08` (incremental
   reindex) and `M-10` (evolutionary retrieval) running live, not narrated over a static screenshot —
   this is what [PRD.md §8](PRD.md#8-jury-scoring-alignment) means by "the demo video shows
   wall-clock timings on screen."

### 7.3 Immediately before the live jury session

Repeat §7.2 in full, on the machine that will actually be used in the room (which may not be the
recording machine). Specifically:

1. Warm the process (§7.2 step 2) — do this in the minutes before the jury arrives, not while they
   are watching. A cold first query in front of the jury is exactly the failure `NFR-07`'s
   degradation-matrix testing and this warm-up step both exist to prevent.
2. Re-confirm `AXIOM_OFFLINE=true` if venue wifi is in doubt — a live query that hangs on a socket
   timeout in front of the jury is a worse outcome than a slightly slower but bounded offline
   response.
3. Have the fallback profile (`configs/fast.yaml`, `AXIOM_LLM_ENABLED=false`) one command away. If
   anything about the primary profile misbehaves live, switching to the fallback and stating plainly
   that this is the documented degradation path (`NFR-07`, `US-12`) is a stronger showing than a
   silent failure — the jury scoring rubric rewards a working prototype, and "it degrades gracefully
   exactly as designed" is itself evidence of the technical-depth criterion in
   [PRD.md §8](PRD.md#8-jury-scoring-alignment).
4. Close every other application that might compete for the reference box's 8 cores or 16 GB RAM
   during the live session — a `NFR-03`/`NFR-04` latency budget measured on an otherwise-idle
   reference box is not the same measurement on a laptop also running a video call and a browser
   with forty tabs.

---

## 8. Related documents

| Document | Relationship |
|---|---|
| [Setup.md](Setup.md) | Prerequisites, environment variables, and the verification ladder this document builds on |
| [PRD.md](PRD.md) | Definition of Done (§2.1), the seven preconditions checked in §4.1 |
| [ImplementationPlan.md](ImplementationPlan.md) | `M7` gate this runbook satisfies |
| [Tracker.md](Tracker.md) | `T-220`–`T-230` submission tasks; §6 demo-day readiness checklist |
| [TestPlan.md](TestPlan.md) | §7 manual test script, `M-01`–`M-13`, executed in §7 above |
| [NonGoals.md](NonGoals.md) | `NG-10`, `NG-13` — why this document has no production rollback machinery |
| [Security.md](Security.md) | Trust boundary for the API/UI surfaces this runbook stands up |
