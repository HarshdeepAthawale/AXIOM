# Setup

Clean clone to a running Axiom system on a CPU-only machine, in one sitting.

**Owner:** Parth Deshmukh
**Last updated:** 2026-09-15
**Status:** Draft

Related: [Deployment.md](Deployment.md) (containers, release, demo day), [API.md](API.md) (HTTP + CLI contracts), [Architecture.md](Architecture.md), [Evaluation.md](Evaluation.md), [OpenQuestions.md](OpenQuestions.md).

---

## 1. Prerequisites

| Requirement | Value | Notes |
|---|---|---|
| Python | 3.11 (3.11-3.12 supported) | 3.13 **untested** — `faiss-cpu` and `llama-cpp-python` wheels lag. Do not use it. |
| git | ≥ 2.34 | Required at runtime, not just for cloning: P1 incremental reindex shells out to `git diff --name-status`. |
| Free disk | ~8 GB | ~3.5 GB HF model cache, ~1.6 GB ONNX artifacts, ~1.0 GB GGUF, ~0.1 GB dataset, ~1.5 GB indices + blobs, rest headroom. |
| RAM | 16 GB recommended, 8 GB minimum | 8 GB works if you set `AXIOM_EMBEDDING_BATCH_SIZE=16` and `AXIOM_LLM_ENABLED=false`. Budgets in [_CONTRACT.md](_CONTRACT.md) §7 assume the 16 GB / 8-core reference box. |
| GPU | **Not required and not used** | The judging hardware is CPU-only. There is no CUDA code path anywhere in `src/axiom/`. See §3. |
| C toolchain | Only if a wheel is missing | `build-essential` (Linux) / Xcode CLT (macOS) / MSVC Build Tools (Windows). Needed only when `tree-sitter` or `llama-cpp-python` fall back to source builds. |
| `uv` | ≥ 0.4 | Primary dependency manager. `pip` fallback documented in §4.2. |
| Network | Needed once | After the pre-download step in §5.4 the whole system runs offline. Assume venue wifi fails. |

Reference box for every number in this document: 8 physical cores, 16 GB RAM, NVMe SSD, no GPU.

---

## 2. Platform Notes

The team is on three different platforms. Read your row before running anything.

| Platform | Status | What is different |
|---|---|---|
| Linux x86_64 (Ubuntu 22.04/24.04) | Primary / CI / judge reference | Default PyPI `torch` is a **CUDA** build (~2.5 GB of unusable nvidia wheels). §3 is mandatory here. |
| macOS arm64 (M-series) | Supported dev target | PyPI `torch` is already CPU-only. `faiss-cpu` publishes arm64 wheels. `llama-cpp-python` builds Metal support by default — disable it (below). |
| Windows 11 x86_64 | Supported dev target | Activation script path differs. Long-path support must be on for HF cache. |
| Windows 11 **arm64** | Degraded — use WSL2 | No `faiss-cpu` wheel for win-arm64. See §2.3. |

### 2.1 Linux

```bash
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev git build-essential
```

- `faiss-cpu` ships manylinux x86_64 wheels — no source build.
- `tree-sitter-javascript` ships wheels; no grammar compilation step is required. If you see a source build, you are on an unsupported interpreter (§7, row "tree-sitter grammar not built").

### 2.2 macOS arm64

```bash
xcode-select --install          # once, for the C toolchain
brew install python@3.11 git
```

- PyPI `torch` for macOS is CPU/MPS only. The `--index-url .../whl/cpu` flag in §3 is a **harmless no-op** here — run the same command anyway so the team has one procedure.
- We do **not** use MPS. Build `llama-cpp-python` without Metal so behaviour matches the judge box:
  ```bash
  CMAKE_ARGS="-DGGML_METAL=OFF -DGGML_ACCELERATE=ON" uv pip install llama-cpp-python --no-binary llama-cpp-python
  ```
- `faiss-cpu` has arm64 macOS wheels from 1.8.0 onward. If resolution picks something older, pin `faiss-cpu>=1.8.0`.

### 2.3 Windows

```powershell
winget install Python.Python.3.11
winget install Git.Git
```

- **Activation differs.** Every command in this doc that says `source .venv/bin/activate` is, on Windows:
  ```powershell
  .venv\Scripts\activate          # PowerShell / cmd
  ```
  In Git Bash it is `source .venv/Scripts/activate` — note `Scripts`, not `bin`. This is the single most common "it works on my machine" failure in the team.
- **Enable long paths** before the first model download, or the HF cache will fail on deeply nested blob names:
  ```powershell
  New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
    -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force
  ```
- **Windows on arm64 (Snapdragon X-series): `faiss-cpu` has no wheel.** Do not try to build it. Use WSL2 with Ubuntu 24.04 and follow the Linux path:
  ```powershell
  wsl --install -d Ubuntu-24.04
  ```
  Keep the repo inside the WSL filesystem (`~/axiom`), not on `/mnt/c` — the `/mnt/c` bridge makes index writes ~5x slower and breaks `git diff` mtime assumptions.
- Paths inside [_CONTRACT.md](_CONTRACT.md) §4 (`ChunkLocation.file_path`) are **always POSIX-separated**, even when indexing on Windows. The chunker normalises separators; never write a backslash into a manifest by hand.

---

## 3. Install CPU torch FIRST (mandatory, all platforms)

Run this **before** any other dependency resolution, in the activated environment:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Why this is a hard rule:

- On Linux, `pip install torch` from default PyPI pulls the **CUDA** build plus ~2.5 GB of `nvidia-*` wheels. On CPU-only judging hardware those bytes are dead weight, they slow the Docker build, and they inflate the image past anything reasonable to ship.
- On macOS the PyPI wheel is already CPU-only, so the `--index-url` flag changes nothing. One command for everyone means no per-OS branch to get wrong.
- Installing torch first pins it, so the later `uv sync` / `pip install -r requirements.txt` resolution will **not** replace it with a CUDA variant.

Verify immediately — if this prints a CUDA suffix, stop and redo:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expected: 2.4.1+cpu False
# WRONG:    2.4.1+cu121 False   <- CUDA wheel, uninstall and repeat
```

Torch is used only for the one-time ONNX export (§6) and as a fallback embedding runtime when `AXIOM_EMBEDDING_BACKEND=torch`. The steady-state hot path is ONNX Runtime.

---

## 4. Install

### 4.1 Primary path — `uv`

```bash
git clone https://github.com/Incognito/axiom.git
cd axiom

uv venv --python 3.11
source .venv/bin/activate          # Windows: .venv\Scripts\activate

uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # §3, first
uv sync --frozen                    # resolves from committed uv.lock
```

`--frozen` fails loudly if `uv.lock` and `pyproject.toml` have drifted. That is the point: CI, the demo box, and the judge box must all resolve to identical versions.

To add a dependency (do not hand-edit the lock):

```bash
uv add bm25s
uv lock
git add pyproject.toml uv.lock && git commit -m "deps: add bm25s"
```

### 4.2 Fallback path — stdlib venv + pip

For anyone who cannot install `uv`, or for a minimal container layer:

```bash
git clone https://github.com/Incognito/axiom.git
cd axiom

python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip

pip install torch --index-url https://download.pytorch.org/whl/cpu      # §3, first
pip install -r requirements.txt
pip install -e .                    # src-layout: makes `import axiom` and `axiom` CLI work
```

`requirements.txt` is generated from the lock, never hand-written:

```bash
uv export --frozen --no-hashes --format requirements-txt > requirements.txt
```

### 4.3 Dev extras

```bash
uv sync --frozen --extra dev        # ruff, mypy, pytest, pytest-cov, types-*
pre-commit install                   # optional; runs ruff on staged files
```

### 4.4 MTEB v2 is required — not v1

`mteb>=2.0` is pinned. The eval adapter in `src/axiom/eval/mteb_adapter.py` targets three APIs that **do not exist in MTEB v1**:

| API used | Introduced in |
|---|---|
| `mteb.evaluate(model, tasks, ...)` | v2 (v1 used `MTEB(tasks=...).run(model)`) |
| `mteb.models.abs_encoder.AbsEncoder` | v2 (v1 had `mteb.encoder_interface.Encoder`) |
| `mteb.abstasks.SearchProtocol` | v2 only |

Check before you debug anything eval-related:

```bash
python -c "import mteb; print(mteb.__version__); print(hasattr(mteb,'evaluate'))"
# expected: 2.x.y  True
```

If you see `1.x` or `False`, your environment is wrong, not the code. `pip install -U 'mteb>=2.0'`.

---

## 5. Models and Dataset

### 5.1 What gets downloaded

Model roles and fallbacks are locked in [_CONTRACT.md](_CONTRACT.md) §2. Sizes are on-disk after download.

| Role | Repo | Source | Approx size | Fallback | Fallback size |
|---|---|---|---|---|---|
| Dense embedder | `Qwen/Qwen3-Embedding-0.6B` | HF Hub | 1.2 GB (bf16 safetensors) | `sentence-transformers/all-MiniLM-L6-v2` | 90 MB |
| Cross-encoder reranker | `BAAI/bge-reranker-v2-m3` | HF Hub | 2.3 GB (fp32 safetensors) | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 90 MB |
| Query LLM (GGUF) | `Qwen/Qwen2.5-1.5B-Instruct-GGUF`, file `qwen2.5-1.5b-instruct-q4_k_m.gguf` | HF Hub | 1.1 GB | heuristic rule engine, no download | 0 |
| JS grammar | `tree-sitter-javascript` | PyPI wheel | 3 MB | regex identifier extraction | 0 |

Post-export INT8 ONNX artifacts add ~0.6 GB (embedder) + ~0.6 GB (reranker). Budget ~7 GB for models in total, including the HF cache copy that the export reads from.

### 5.2 Where weights cache

Everything obeys `HF_HOME`. Set it explicitly so the demo box, the container, and your laptop agree:

```bash
export HF_HOME="$PWD/data/hf"
```

Resulting layout:

```
data/
├── hf/hub/models--Qwen--Qwen3-Embedding-0.6B/...
├── hf/hub/models--BAAI--bge-reranker-v2-m3/...
├── hf/datasets/CoIR-Retrieval___apps/...
├── models/
│   ├── onnx/qwen3-embedding-0.6b-int8/model.onnx
│   ├── onnx/bge-reranker-v2-m3-int8/model.onnx
│   └── gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf
└── datasets/            # any locally staged corpora
```

`data/` is gitignored ([_CONTRACT.md](_CONTRACT.md) §3). Never commit weights.

### 5.3 Download

```bash
huggingface-cli download Qwen/Qwen3-Embedding-0.6B
huggingface-cli download BAAI/bge-reranker-v2-m3
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --include "qwen2.5-1.5b-instruct-q4_k_m.gguf" \
  --local-dir data/models/gguf --local-dir-use-symlinks False
```

Downloads resume. If one is interrupted, re-run the same command — do **not** delete the cache (§7).

### 5.4 Pre-download for an offline demo (do this the day before)

Venue wifi is unreliable and the demo must not depend on it. Pull the fallbacks too, so a primary-model failure degrades instead of dying:

```bash
export HF_HOME="$PWD/data/hf"
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2
huggingface-cli download cross-encoder/ms-marco-MiniLM-L-6-v2
python -c "import mteb; mteb.get_task('AppsRetrieval').load_data()"
```

Then flip the environment to hard-offline and prove it still works:

```bash
export HF_HUB_OFFLINE=1
export AXIOM_OFFLINE=true
axiom search "how is the input normalized" --top-k 5
```

`AXIOM_OFFLINE=true` makes any attempted network fetch raise `MODEL_UNAVAILABLE` immediately (see [API.md](API.md) §6) instead of hanging on a socket timeout in front of the jury.

### 5.5 Dataset — CoIR AppsRetrieval

**Read this before you touch the eval path.** `CoIR-Retrieval/apps` is the APPS dataset: **English competitive-programming problem statements retrieving Python solutions**. Each corpus entry is a single-file, standalone program. There is no cross-file call graph, no import graph, and nothing for the structural signal to traverse.

The **JavaScript** constraint in the problem statement applies to the **live-demo codebase**, not to the benchmark. These are two different corpora with two different index trees and two different config profiles (§7.0):

| | Benchmark corpus | Demo corpus |
|---|---|---|
| Content | 8,765 standalone Python snippets | multi-version JavaScript repository |
| Used for | NDCG@10 / MRR score, screening round | live jury demo, P1 version switch, Bonus evolutionary |
| Profile | `eval` — structural signal **off**, dense-weighted | `demo` — all three signals on |
| Index tree | `.axiom-apps/` | `.axiom-demo/` |

Do not "fix" low structural recall on APPS. There is nothing structural in it; `eval.yaml` disables that signal deliberately. See [Decisions.md](Decisions.md) and [TestPlan.md](TestPlan.md).

| Split | Entries | What we do with it |
|---|---|---|
| Corpus | 8,765 Python code snippets | indexed; the retrieval target |
| Queries | 8,765 natural-language queries | problem statements |
| Test relevance pairs | 3,765 | **scored**. Never touched during tuning. |
| Train pairs | 5,000 | **tuning only** — RRF fusion weights and rerank score thresholds |

**Hygiene rule (non-negotiable).** Fusion weights, the rerank cutoff, agent sufficiency thresholds, and any prompt change are tuned against the **5,000 train pairs**. The 3,765 test pairs are read exactly once per candidate configuration, to report a number. Tuning a knob, reading test, and re-tuning is test-set fitting; it invalidates the submitted score. Download both splits — the train pairs are not optional.

Source: `CoIR-Retrieval/apps` on the HF Hub, loaded through MTEB so the qrels wiring matches what the scorer expects.

```bash
export HF_HOME="$PWD/data/hf"
python - <<'PY'
import mteb
task = mteb.get_task("AppsRetrieval")
task.load_data()                     # pulls corpus, queries, and both qrel splits
print(task.metadata.name, "ok")
PY
```

Disk footprint ~85 MB raw parquet, ~110 MB after the Arrow cache is written. Local cache path: `data/hf/datasets/CoIR-Retrieval___apps/`.

Do not hand-download the parquet files and point the adapter at them — MTEB v2 owns split selection and qrel alignment, and a hand-rolled loader is how you silently evaluate on the wrong split.

**FAISS index kind on the benchmark: `IndexFlatIP`, always.** 8,765 vectors is far below the `AXIOM_FAISS_IVF_THRESHOLD` of 50,000, so the dense index is exact brute force — and at that scale it is also instant (~6 ms for a 1024-dim query over 8,765 vectors on one core). IVF-PQ is a lossy approximation that exists for large demo repos only. Forcing it on the benchmark costs recall for no measurable latency win. If you see `index_kind: "ivf_pq"` in an APPS manifest, someone "optimised" it; rebuild.

---

## 6. ONNX Export and INT8 Quantisation

The hot path runs INT8 ONNX on `onnxruntime` CPU. Export is a one-time step per model; the artifacts are gitignored and rebuilt or copied to each box.

### 6.1 Export

```bash
export HF_HOME="$PWD/data/hf"

# Dense embedder: HF -> ONNX (fp32) -> INT8 dynamic
optimum-cli export onnx \
  --model Qwen/Qwen3-Embedding-0.6B \
  --task feature-extraction \
  --opset 17 \
  data/models/onnx/qwen3-embedding-0.6b-fp32/

optimum-cli onnxruntime quantize \
  --onnx_model data/models/onnx/qwen3-embedding-0.6b-fp32/ \
  --avx512_vnni \
  -o data/models/onnx/qwen3-embedding-0.6b-int8/

# Cross-encoder reranker: HF -> ONNX (fp32) -> INT8 dynamic
optimum-cli export onnx \
  --model BAAI/bge-reranker-v2-m3 \
  --task text-classification \
  --opset 17 \
  data/models/onnx/bge-reranker-v2-m3-fp32/

optimum-cli onnxruntime quantize \
  --onnx_model data/models/onnx/bge-reranker-v2-m3-fp32/ \
  --avx512_vnni \
  -o data/models/onnx/bge-reranker-v2-m3-int8/
```

Notes:

- `--opset 17` is deliberate. `onnxruntime` 1.18 supports up to opset 20, but opset 17 is the highest that the `--avx512_vnni` dynamic quantiser handles cleanly for these architectures. Raising it produces the opset mismatch in §7.
- On arm64 (macOS, WSL on Snapdragon) drop `--avx512_vnni`; the quantiser falls back to portable QOperator kernels. Throughput drops roughly 25%, accuracy is unchanged.
- Export peaks at ~6 GB RSS for the reranker. On an 8 GB box, export the two models in separate processes, not in one script.
- Delete the `-fp32/` directories after quantising if disk is tight; they are only inputs.

### 6.2 Verify the artifact

```bash
python - <<'PY'
import onnxruntime as ort, numpy as np
p = "data/models/onnx/qwen3-embedding-0.6b-int8/model.onnx"
s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
print("inputs :", [(i.name, i.shape) for i in s.get_inputs()])
print("outputs:", [(o.name, o.shape) for o in s.get_outputs()])
ids = np.ones((1, 16), dtype=np.int64)
out = s.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0]
print("hidden :", out.shape[-1])
PY
```

Expected:

```
inputs : [('input_ids', [...]), ('attention_mask', [...])]
outputs: [('last_hidden_state', [...])]
hidden : 1024
```

`hidden` **must** be 1024 for Qwen3-0.6B (384 for the MiniLM fallback), matching `VersionManifest.embedding_dim` in [Schema.md](Schema.md). A mismatch here means every index you build afterwards is unloadable.

Sanity-check quantisation actually happened — the INT8 file should be roughly 1/4 the fp32 size:

```bash
du -sh data/models/onnx/qwen3-embedding-0.6b-fp32 data/models/onnx/qwen3-embedding-0.6b-int8
# ~2.3G   fp32
# ~0.6G   int8
```

---

## 7. Environment Variables

All settings are `pydantic-settings` v2 fields with the `AXIOM_` prefix, layered over the YAML profile named by `AXIOM_PROFILE`. Precedence, lowest to highest: profile YAML in `configs/` → `.env` → process environment → CLI flag.

### 7.1 Core

| Variable | Type | Default | Effect |
|---|---|---|---|
| `AXIOM_PROFILE` | str | `default` | Loads `configs/<profile>.yaml`. Valid: `default`, `fast`, `accurate`, `eval`. |
| `AXIOM_INDEX_ROOT` | path | `.axiom` | Root of the on-disk index tree ([_CONTRACT.md](_CONTRACT.md) §6). |
| `AXIOM_DATA_ROOT` | path | `data` | Models, datasets, HF cache parent. |
| `AXIOM_LOG_LEVEL` | str | `INFO` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR`. |
| `AXIOM_LOG_FORMAT` | str | `console` | `console` for humans, `json` for CI and the demo box. |
| `AXIOM_NUM_THREADS` | int | `0` | ONNX Runtime intra-op threads and FAISS `omp_set_num_threads`. `0` = physical core count. Set to 4 on a laptop to keep the UI responsive while indexing. |
| `AXIOM_SEED` | int | `42` | Seeds numpy/random for IVF training and any sampling. Deterministic eval depends on it. |
| `AXIOM_OFFLINE` | bool | `false` | Refuse all network access; also exports `HF_HUB_OFFLINE=1` to child processes. Use for the demo. |

### 7.2 Models

| Variable | Type | Default | Effect |
|---|---|---|---|
| `AXIOM_EMBEDDING_MODEL` | str | `Qwen/Qwen3-Embedding-0.6B` | HF repo id of the dense embedder. Set to `sentence-transformers/all-MiniLM-L6-v2` to use the fallback. |
| `AXIOM_EMBEDDING_BACKEND` | str | `onnx` | `onnx` (INT8, production) or `torch` (fp32, debugging only — ~4x slower). |
| `AXIOM_EMBEDDING_ONNX_PATH` | path | `data/models/onnx/qwen3-embedding-0.6b-int8/model.onnx` | Overrides the derived artifact path. |
| `AXIOM_EMBEDDING_DIM` | int | `1024` | Asserted against the loaded model at startup; a mismatch is fatal, not a warning. |
| `AXIOM_EMBEDDING_BATCH_SIZE` | int | `64` | Chunks per forward pass during indexing. Drop to `16` on an 8 GB box. |
| `AXIOM_EMBEDDING_MAX_TOKENS` | int | `512` | Truncation length for chunk text. |
| `AXIOM_RERANKER_MODEL` | str | `BAAI/bge-reranker-v2-m3` | HF repo id of the cross-encoder. |
| `AXIOM_RERANKER_ENABLED` | bool | `true` | `false` returns the raw RRF ordering. Costs roughly 5-8 NDCG@10 points; useful to isolate a regression. |
| `AXIOM_RERANKER_ONNX_PATH` | path | `data/models/onnx/bge-reranker-v2-m3-int8/model.onnx` | Overrides the derived artifact path. |
| `AXIOM_RERANKER_TIMEOUT_MS` | int | `2500` | Per-query rerank wall clock. On expiry the RRF order is returned and `RERANKER_TIMEOUT` is surfaced as a warning. |

### 7.3 Agent and LLM

| Variable | Type | Default | Effect |
|---|---|---|---|
| `AXIOM_LLM_ENABLED` | bool | `true` | `false` swaps the GGUF query LLM for the heuristic rule engine. **The whole pipeline must pass its tests with this set to `false`** ([_CONTRACT.md](_CONTRACT.md) §2). |
| `AXIOM_LLM_MODEL_PATH` | path | `data/models/gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf` | GGUF weights for `llama-cpp-python`. |
| `AXIOM_LLM_MAX_TOKENS` | int | `512` | Hard output cap. The LLM never reads code, only classifies/expands/decomposes/evaluates. |
| `AXIOM_LLM_CONTEXT` | int | `2048` | `n_ctx` for llama.cpp. |
| `AXIOM_AGENT_ENABLED` | bool | `true` | `false` runs a single static pass: retrieve → fuse → rerank → return. |
| `AXIOM_AGENT_MAX_PASSES` | int | `2` | Maximum refinement passes. Locked at 2 for the hackathon. |
| `AXIOM_AGENT_WALL_CLOCK_MS` | int | `5000` | Hard per-query budget. On expiry the best results so far are returned with `AGENT_BUDGET_EXCEEDED` as a warning. |
| `AXIOM_AGENT_SUFFICIENCY_TOP1` | float | `0.35` | Refine if top-1 rerank score is below this. |
| `AXIOM_AGENT_SUFFICIENCY_FLOOR` | float | `0.20` | Score floor used by the "fewer than N results above floor" trigger. |
| `AXIOM_AGENT_SUFFICIENCY_MIN_RESULTS` | int | `3` | The N above. |

### 7.4 Retrieval and fusion

| Variable | Type | Default | Effect |
|---|---|---|---|
| `AXIOM_RRF_K` | int | `60` | RRF constant. Locked; do not tune it without an [ADR](Decisions.md). |
| `AXIOM_DENSE_TOP_K` | int | `100` | Dense candidate width. |
| `AXIOM_SPARSE_TOP_K` | int | `100` | BM25 candidate width. |
| `AXIOM_STRUCTURAL_TOP_K` | int | `50` | Structural candidate width. |
| `AXIOM_FUSION_TOP_N` | int | `25` | Candidates handed to the reranker. |
| `AXIOM_TOP_K_DEFAULT` | int | `10` | Default result count. |
| `AXIOM_TOP_K_MAX` | int | `100` | Ceiling, enforced by the API ([API.md](API.md) §7). |
| `AXIOM_FAISS_IVF_THRESHOLD` | int | `50000` | Vectors at or above this use `IndexIVFPQ`; below, `IndexFlatIP`. |
| `AXIOM_CHUNK_MIN_TOKENS` | int | `16` | Chunks smaller than this merge into their parent. |
| `AXIOM_CHUNK_TARGET_TOKENS` | int | `512` | Upper target; oversized functions split at statement boundaries with 1-statement overlap. |

### 7.5 Versioning and evolutionary

| Variable | Type | Default | Effect |
|---|---|---|---|
| `AXIOM_ACTIVE_VERSION` | str \| null | `null` | Overrides the active version in `registry.json`. `null` means "use the registry". |
| `AXIOM_EVOLUTIONARY_ENABLED` | bool | `false` | Enables cross-version search and `SnippetFamily` grouping. Off by default because it widens candidate sets. |
| `AXIOM_DEDUPE_COSINE` | float | `0.95` | Family membership threshold within the same `symbol` + `file_path`. |
| `AXIOM_STABILITY_BONUS` | float | `0.10` | `final = base * (1 + bonus * stability)` for families in ≥2 versions. |

### 7.6 Services

| Variable | Type | Default | Effect |
|---|---|---|---|
| `AXIOM_API_HOST` | str | `127.0.0.1` | Bind address for `axiom serve`. Use `0.0.0.0` inside Docker. |
| `AXIOM_API_PORT` | int | `8000` | API port. |
| `AXIOM_UI_PORT` | int | `8501` | Streamlit port. |
| `AXIOM_API_BASE_URL` | str | `http://127.0.0.1:8000` | Where the Streamlit UI looks for the API. |

### 7.7 Third-party variables we also set

| Variable | Value we use | Why |
|---|---|---|
| `HF_HOME` | `./data/hf` | Keeps the model cache inside the repo tree so it is easy to copy to the demo box. |
| `HF_HUB_OFFLINE` | `1` for demo | Fail fast instead of hanging when wifi dies. |
| `TOKENIZERS_PARALLELISM` | `false` | Silences the fork warning and avoids thread oversubscription during batched embedding. |
| `OMP_NUM_THREADS` | mirrors `AXIOM_NUM_THREADS` | FAISS and ONNX Runtime both read it. |

### 7.8 `.env.example`

Copy to `.env` (gitignored) and edit:

```dotenv
# ---- core ----
AXIOM_PROFILE=default
AXIOM_INDEX_ROOT=.axiom
AXIOM_DATA_ROOT=data
AXIOM_LOG_LEVEL=INFO
AXIOM_LOG_FORMAT=console
AXIOM_NUM_THREADS=0
AXIOM_SEED=42
AXIOM_OFFLINE=false

# ---- models ----
AXIOM_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
AXIOM_EMBEDDING_BACKEND=onnx
AXIOM_EMBEDDING_DIM=1024
AXIOM_EMBEDDING_BATCH_SIZE=64
AXIOM_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
AXIOM_RERANKER_ENABLED=true
AXIOM_RERANKER_TIMEOUT_MS=2500

# ---- agent ----
AXIOM_LLM_ENABLED=true
AXIOM_LLM_MODEL_PATH=data/models/gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf
AXIOM_AGENT_ENABLED=true
AXIOM_AGENT_MAX_PASSES=2
AXIOM_AGENT_WALL_CLOCK_MS=5000

# ---- retrieval ----
AXIOM_RRF_K=60
AXIOM_DENSE_TOP_K=100
AXIOM_SPARSE_TOP_K=100
AXIOM_STRUCTURAL_TOP_K=50
AXIOM_FUSION_TOP_N=25
AXIOM_TOP_K_DEFAULT=10

# ---- versioning ----
AXIOM_EVOLUTIONARY_ENABLED=false
AXIOM_DEDUPE_COSINE=0.95
AXIOM_STABILITY_BONUS=0.10

# ---- services ----
AXIOM_API_HOST=127.0.0.1
AXIOM_API_PORT=8000
AXIOM_UI_PORT=8501

# ---- third party ----
HF_HOME=./data/hf
TOKENIZERS_PARALLELISM=false
```

---

## 8. Verification Ladder

Run these in order. Each one is cheap and isolates a different layer; stop at the first failure and go to §9.

### Rung 1 — imports and native libs (~2 s)

```bash
python - <<'PY'
import torch, onnxruntime, faiss, bm25s, tree_sitter, tree_sitter_javascript, mteb
print("torch      ", torch.__version__)
print("onnxruntime", onnxruntime.__version__, onnxruntime.get_available_providers())
print("faiss      ", faiss.__version__)
print("mteb       ", mteb.__version__, hasattr(mteb, "evaluate"))
PY
```

Expected:

```
torch       2.4.1+cpu
onnxruntime 1.18.1 ['CPUExecutionProvider']
faiss       1.8.0
mteb        2.1.0 True
```

`+cpu`, a providers list with **only** `CPUExecutionProvider`, and `True` for MTEB v2. Anything else, stop.

### Rung 2 — package wiring and config (~1 s)

```bash
axiom --version
axiom config show --profile default
```

Expected:

```
axiom 0.1.0 (profile=default, embedding=Qwen/Qwen3-Embedding-0.6B, dim=1024, llm=on)

profile              default
index_root           .axiom
embedding_model      Qwen/Qwen3-Embedding-0.6B
embedding_backend    onnx
embedding_dim        1024
reranker_model       BAAI/bge-reranker-v2-m3
reranker_enabled     True
llm_enabled          True
agent_max_passes     2
rrf_k                60
num_threads          8 (auto)
```

`command not found` means you skipped `pip install -e .` or are in the wrong venv.

### Rung 3 — unit tests (~40 s)

```bash
pytest -q tests/ -x --timeout=120
```

Expected tail:

```
187 passed, 3 skipped in 38.42s
```

Skips are the LLM-dependent tests when no GGUF is present — acceptable. Any failure here is a code problem, not a setup problem; see [TestPlan.md](TestPlan.md).

### Rung 4 — smoke index over the fixture repo (~25 s)

`tests/fixtures/mini_repo/` is a 14-file JavaScript toy repo committed to the repo specifically so setup can be verified without the real corpus.

```bash
axiom index tests/fixtures/mini_repo --version-id smoke --index-root /tmp/axiom-smoke
```

Expected:

```
[00:00] discover   14 files (.js)
[00:01] chunk      97 chunks  (function=71 method=14 class=8 module=4)
[00:09] embed      97 vectors dim=1024 backend=onnx batch=64
[00:10] sparse     bm25s index built, vocab=2,431
[00:11] structural 97 symbols, 168 calls, 22 imports, 19 exports
[00:11] manifest   /tmp/axiom-smoke/index/smoke/manifest.json
done: version=smoke chunks=97 elapsed=11.4s
```

Then confirm the on-disk layout matches [_CONTRACT.md](_CONTRACT.md) §6:

```bash
find /tmp/axiom-smoke -maxdepth 3 -type f -o -maxdepth 3 -type d | sort
# /tmp/axiom-smoke/registry.json
# /tmp/axiom-smoke/blobs
# /tmp/axiom-smoke/index/smoke/manifest.json
# /tmp/axiom-smoke/index/smoke/chunks.jsonl
# /tmp/axiom-smoke/index/smoke/dense.faiss
# /tmp/axiom-smoke/index/smoke/dense.idmap.json
# /tmp/axiom-smoke/index/smoke/sparse.bm25s
# /tmp/axiom-smoke/index/smoke/structural.sqlite
```

### Rung 5 — smoke search returning ranked results (~2 s)

```bash
axiom search "how is user input normalized before dispatch" \
  --version smoke --index-root /tmp/axiom-smoke --top-k 3
```

Expected:

```
query_type=SEMANTIC  passes=1  latency=712ms  reranker=on  llm=on

1. 0.8421  src/utils/normalize.js:10-25  normalizeInput()
   match_reason: dense rank 1, sparse rank 4 - semantic match on "normalize"/"sanitize"
   signals: DENSE=1 SPARSE=4

2. 0.6107  src/agents/dispatch.js:44-71  dispatch()
   match_reason: structural rank 2 - calls normalizeInput() before route()
   signals: DENSE=7 STRUCTURAL=2

3. 0.5533  src/utils/sanitize.js:3-19   stripControlChars()
   match_reason: dense rank 3 - semantic neighbour of normalize
   signals: DENSE=3
```

Three ranked hits with real file paths, line ranges, scores, and a `match_reason` means the full stack works: chunking, all three indices, RRF, and the reranker. You are set up.

Finally, prove the degraded path also works — this is the configuration the judges may hit:

```bash
AXIOM_LLM_ENABLED=false AXIOM_RERANKER_ENABLED=false \
  axiom search "normalize input" --version smoke --index-root /tmp/axiom-smoke --top-k 3
# query_type=SEMANTIC (heuristic)  passes=1  latency=214ms  reranker=off  llm=off
```

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `torch.__version__` ends in `+cu121`; `pip` downloaded ~2.5 GB of `nvidia-*` wheels | Default PyPI torch is the CUDA build on Linux; §3 was skipped or run after `uv sync` | `pip uninstall -y torch` and every `nvidia-*` package, then re-run §3 **first**, then `uv sync --frozen`. Check with `pip list \| grep -i nvidia` — must be empty. |
| `ImportError: libfaiss... cannot open shared object file` or `No matching distribution found for faiss-cpu` on arm64 | No `faiss-cpu` wheel for your platform/interpreter (Windows arm64 has none; macOS arm64 needs ≥1.8.0) | macOS: `uv pip install 'faiss-cpu>=1.8.0'`. Windows arm64: switch to WSL2 Ubuntu (§2.3). Never attempt a source build during the build window. |
| `TypeError: Language.__init__() takes 2 positional arguments but 3 were given`, or setuptools tries to compile a grammar | `tree-sitter` < 0.22 API, or a source install because no wheel matched your Python | Pin `tree-sitter>=0.22` and `tree-sitter-javascript>=0.21`; load with `Language(tree_sitter_javascript.language())`. Confirm you are on Python 3.11/3.12. Meanwhile the chunker degrades to the regex identifier fallback — structural recall drops, nothing crashes. |
| `429 Too Many Requests` from `huggingface.co`, or downloads hang at 0 B | HF anonymous rate limit, or you are offline | `huggingface-cli login` with a read token, or wait 5 minutes. For the demo, pre-download (§5.4) and set `HF_HUB_OFFLINE=1` + `AXIOM_OFFLINE=true`. |
| Model download dies partway; re-running seems to restart from zero | Interrupted transfer left `.incomplete` blobs; you deleted the cache and lost resume state | Just re-run the same `huggingface-cli download` — it resumes from `.incomplete`. If genuinely corrupt, delete only the one `models--<org>--<name>/blobs/<sha>.incomplete` file, not `HF_HOME`. |
| `onnxruntime` raises `Unsupported model IR version` / `opset 21 not supported` | Exported with a newer opset than `onnxruntime` 1.18 supports, or `--opset` omitted so optimum picked the latest | Re-export with `--opset 17` (§6.1). If you must use a higher opset, upgrade `onnxruntime` and record it as an [ADR](Decisions.md). |
| Indexing killed by the OOM reaper, or RSS climbs past 8 GB during `axiom index` | Embedding batch too large for the box; or `AXIOM_EMBEDDING_BACKEND=torch` (fp32) instead of `onnx` | `AXIOM_EMBEDDING_BATCH_SIZE=16`, `AXIOM_NUM_THREADS=4`, and confirm `AXIOM_EMBEDDING_BACKEND=onnx`. Budget is ≤ 4 GB peak RSS per [_CONTRACT.md](_CONTRACT.md) §7. |
| `AttributeError: module 'mteb' has no attribute 'evaluate'`, or `ImportError: cannot import name 'AbsEncoder'` | MTEB **v1** installed. v1 has `MTEB(...).run()` and `Encoder`, not `mteb.evaluate`, `AbsEncoder`, or `SearchProtocol` | `pip install -U 'mteb>=2.0'`, verify with §4.4. Never patch the adapter to the v1 API. |
| `[Errno 98] Address already in use` on 8000 or 8501 | A previous `axiom serve` / `axiom ui` is still running | `lsof -ti:8000 \| xargs kill` (Linux/macOS) or `netstat -ano \| findstr :8000` then `taskkill /PID <pid> /F` (Windows). Or run on another port: `axiom serve --port 8010`. |
| Search returns stale results, or `KeyError` on a chunk id after you changed the chunker | Index on disk was built by older code; chunk ids are content-derived and shifted | `axiom index <repo> --version-id <id> --force` to rebuild, or delete `$AXIOM_INDEX_ROOT/index/<version_id>/`. The manifest records `embedding_model`; a model change always requires a rebuild. |
| Startup aborts with `embedding_dim mismatch: manifest=1024 model=384` | The index was built with Qwen3-0.6B but `AXIOM_EMBEDDING_MODEL` now points at MiniLM (or vice versa) | Either restore the original model or rebuild the index. Dimensions are not convertible. |
| `axiom: command not found` after a successful install | `pip install -e .` skipped (pip fallback path), or the venv is not activated | Activate (`.venv\Scripts\activate` on Windows — §2.3), then `pip install -e .`. Confirm with `which axiom` / `where axiom`. |
| Streamlit UI loads but every query shows "API unreachable" | `axiom serve` not running, or `AXIOM_API_BASE_URL` points at the wrong host/port | Start the API first, then `curl -s localhost:8000/v1/health`. Inside Docker the UI must use the compose service name, not `127.0.0.1` — see [Deployment.md](Deployment.md) §3. |
| First query after startup takes 20-40 s, later ones are fast | Cold model load: ONNX session construction + GGUF mmap happen lazily on first use | Expected. Warm the process before the demo: `curl -s localhost:8000/v1/health?warm=true`. Never let the jury trigger the cold path — see the demo runbook in [Deployment.md](Deployment.md) §7. |
| `git diff` based reindex reports every file as modified | Repo cloned with `core.autocrlf=true` (Windows), so line endings differ from the indexed version | `git config core.autocrlf input` and re-clone, or reindex once to reset the baseline `file_hashes` in the manifest. |

---

## 10. Next Steps

| You want to | Go to |
|---|---|
| Understand the pipeline you just installed | [Architecture.md](Architecture.md) |
| Know the exact field names in results | [Schema.md](Schema.md) |
| Call the API or the CLI | [API.md](API.md) |
| Build a container, cut the release, or run the demo | [Deployment.md](Deployment.md) |
| Reproduce the NDCG@10 number | [Evaluation.md](Evaluation.md) |
| Report something this doc did not cover | [OpenQuestions.md](OpenQuestions.md), as a new `OQ-##` |
