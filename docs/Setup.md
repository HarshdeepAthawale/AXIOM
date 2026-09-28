# Setup

Clean clone to a running Axiom system on a CPU-only machine, in one sitting.

**Owner:** Parth Deshmukh
**Last updated:** 2026-09-23
**Status:** Draft

Related: [Deployment.md](Deployment.md) (containers, release, demo day), [API.md](API.md) (HTTP + CLI contracts), [Design.md](Design.md) (architecture), [TestPlan.md](TestPlan.md) (evaluation protocol), [_CONTRACT.md](_CONTRACT.md) (locked facts), [OpenQuestions.md](OpenQuestions.md).

---

## 1. Prerequisites

| Requirement | Value | Notes |
|---|---|---|
| Python | 3.11 (3.11-3.12 supported) | 3.13 **untested** — `faiss-cpu` and `llama-cpp-python` wheels lag. Do not use it. |
| git | ≥ 2.34 | Required at runtime, not just for cloning: P1 incremental reindex shells out to `git diff --name-status`. |
| Free disk | ~16 GB during setup, ~9 GB steady state | See the arithmetic in §5.1. Peak is during ONNX export, which needs the HF cache *and* the fp32 intermediate *and* the INT8 output on disk at once. |
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
- `tree-sitter-javascript` ships wheels; no grammar compilation step is required. If you see a source build, you are on an unsupported interpreter (§9, the tree-sitter row).

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
  Keep the repo inside the WSL filesystem (`~/AXIOM`), not on `/mnt/c` — the `/mnt/c` bridge makes index writes ~5x slower and breaks `git diff` mtime assumptions.
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

Torch is used only for the one-time ONNX export (§6), and transitively by `mteb` in the `[eval]` extra. **There is no torch embedding runtime** — `AXIOM_EMBEDDING_BACKEND` is not implemented (§7.1). The hot path is ONNX Runtime, and below it the degradation ladder, never torch.

---

## 4. Install

### 4.1 Primary path — `uv`

```bash
git clone https://github.com/HarshdeepAthawale/AXIOM.git
cd AXIOM

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
git clone https://github.com/HarshdeepAthawale/AXIOM.git
cd AXIOM

python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip

pip install torch --index-url https://download.pytorch.org/whl/cpu      # §3, first
pip install -r requirements.txt
pip install -e .                    # src-layout: makes `import axiom` and the `axiom` CLI work
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

Model roles are locked in [_CONTRACT.md](_CONTRACT.md) §2. Sizes are on-disk after download.

| Role | Repo | Source | Download | Fallback | Fallback download |
|---|---|---|---|---|---|
| Dense embedder | `Qwen/Qwen3-Embedding-0.6B` | HF Hub | 1.2 GB (bf16 safetensors) | `sentence-transformers/all-MiniLM-L6-v2` | 90 MB |
| Cross-encoder reranker | `BAAI/bge-reranker-v2-m3` | HF Hub | 2.3 GB (fp32 safetensors) | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 90 MB |
| Query LLM (GGUF) | `Qwen/Qwen2.5-1.5B-Instruct-GGUF`, file `qwen2.5-1.5b-instruct-q4_k_m.gguf` | HF Hub | 1.1 GB | heuristic rule engine, no download | 0 |
| JS grammar | `tree-sitter-javascript` | PyPI wheel | 3 MB | regex identifier extraction | 0 |

**The reranker row is profile-split** ([_CONTRACT.md §2](_CONTRACT.md#2-model-stack-locked-cpu-only)):
`eval.yaml` keeps `bge-reranker-v2-m3` as primary and buys the cost back by narrowing the candidate
chain; `demo.yaml` promotes `ms-marco-MiniLM-L-6-v2` to primary, because latency is what the jury
watches. Both are downloaded — neither is optional.

#### Disk arithmetic — do this sum before you start, not halfway through the export

`NFR-12`'s ceiling is derived from this table, not asserted independently of it.

| | Primary profile | Fallback profile (`configs/fast.yaml`) |
|---|---|---|
| Downloaded (HF cache) | 1.2 + 2.3 + 1.1 + 0.003 = **~4.6 GB** | 0.09 + 0.09 + 0.003 = **~0.19 GB** |
| ONNX fp32 intermediates (§6.1, deletable afterwards) | ~2.3 + ~2.3 = ~4.6 GB | ~0.36 GB |
| INT8 ONNX artifacts (kept) | ~0.6 + ~0.6 = ~1.2 GB | ~0.09 GB |
| Dataset cache (§5.5) | ~0.11 GB | ~0.11 GB |
| Indices + blobs, 10k chunks | ~1.5 GB | ~0.6 GB |
| **Transient peak during export** | **~10.4 GB** | ~1.1 GB |
| **Steady state after deleting `-fp32/`** | **~7.4 GB** | **~1.0 GB** |

So: **~4.6 GB downloaded, ~7 GB on disk steady state, ~10.4 GB transient peak** for the primary
profile. The "< 2.5 GB" figure that `NFR-12` used to carry is contradicted by this table's own
first row and is not recoverable by rounding — it was written against a stack that did not include
`bge-reranker-v2-m3`. The fallback profile is the one that genuinely meets a < 500 MB download
promise, and that is the promise worth making to a bandwidth-limited evaluator.

Two ways to shrink the primary profile if the evaluator's disk is tight, in order of preference:
delete the `-fp32/` directories the moment each quantise step finishes (§6.1), which is the 3 GB
saving; and, if a network round-trip is cheaper than local disk, pre-export the INT8 artifacts once
and publish them to an HF repo of our own, so the evaluator pulls ~1.2 GB of INT8 ONNX and never
runs `optimum` at all.

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

Downloads resume. If one is interrupted, re-run the same command — do **not** delete the cache (§9).

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
axiom query "how is the input normalized" --top-k 5
```

`AXIOM_OFFLINE=true` makes any attempted network fetch raise `MODEL_UNAVAILABLE` immediately (see [API.md](API.md) §6) instead of hanging on a socket timeout in front of the jury.

### 5.5 Dataset — CoIR AppsRetrieval

**Read this before you touch the eval path.** `CoIR-Retrieval/apps` is the APPS dataset: **English competitive-programming problem statements retrieving Python solutions**. Each corpus entry is a single-file, standalone program. There is no cross-file call graph, no import graph, and nothing for the structural signal to traverse.

The **JavaScript** constraint in the problem statement applies to the **live-demo codebase**, not to the benchmark. These are two different corpora with two different index trees and two different config profiles (§7.1):

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
- Delete the `-fp32/` directories after quantising if disk is tight; they are only inputs. Per the
  §5.1 arithmetic this is a ~3 GB saving and costs nothing — the INT8 artifact is self-contained.

### 6.2 Pooling and the query-side instruction envelope

**Read this before §6.3. It is the highest-probability silent failure in the build, and it lands
directly on the only scored metric.**

`optimum-cli export onnx --task feature-extraction` exports `last_hidden_state` only — a
`(batch, seq, hidden)` tensor. It does **not** export a pooling head. Pooling is therefore
reimplemented in Python, in `src/axiom/indexing/embedder.py`, and the two models in the stack pool
**differently**. A single shared pooler is itself a bug.

| Model | Pooling | Query-side envelope |
|---|---|---|
| `Qwen/Qwen3-Embedding-0.6B` | **last real token** | `Instruct: {task}\nQuery:{query}` |
| `sentence-transformers/all-MiniLM-L6-v2` | attention-mask-weighted **mean** | none |
| hash rung (no model present) | n/a — bag-of-tokens | none |

What the code actually does, so the spec and the implementation cannot drift apart:

1. **Last-token pooling selects the last *real* token, not the last *slot*.** `OnnxEmbedder._pool`
   computes `lengths = attention_mask.sum(axis=1) - 1` and indexes
   `hidden[arange(batch), lengths]`. This matters: the tokenizer pads **right** (the `tokenizers`
   default), so a naive `last_hidden_state[:, -1, :]` would return the PAD token's state for every
   row shorter than the longest in the batch — which, at `AXIOM_EMBEDDING_BATCH_SIZE=64`, is 63 of
   every 64 chunks, silently, producing well-formed, correctly-dimensioned, near-useless vectors.
   The mask-derived index is equivalent to left-padding plus `[:, -1, :]` and does not require the
   padding direction to be reconfigured.
2. **Mean pooling is mask-weighted**, `(hidden * mask).sum(1) / max(mask.sum(1), 1)` — PAD
   positions contribute nothing and do not dilute the denominator.
3. **The pooler is chosen from the model id**, `"qwen3-embedding" in model_id.lower()`, not from
   configuration, so a degrade from the primary rung to MiniLM switches pooling with it.
4. **The instruction envelope is applied on the QUERY SIDE ONLY.** `Embedder.encode_query(text,
   task)` wraps; `Embedder.encode(texts)` — every index-time path — does not. Wrapping the corpus
   would embed every chunk behind the same prefix and collapse their separation. The wrapping is
   also gated on the model: `needs_query_instruction` is true only for the instruction-tuned
   primary, because prefixing a query for a model that was never trained on the envelope makes
   retrieval worse, not better.
5. **The task string is configuration, not a literal.** `Settings.embedding_query_instruction`,
   overridable as `AXIOM_EMBEDDING_QUERY_INSTRUCTION` and committed to `configs/`, so a reported
   score is reproducible from a SHA.
6. **L2 normalisation happens after pooling**, at the output boundary of every rung, so FAISS inner
   product equals cosine everywhere downstream.

The exact envelope, byte for byte — note there is **no space** after `Query:`:

```text
Instruct: {task}\nQuery:{query}
```

### 6.3 Verify the artifact

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

### 6.4 Verify the *behaviour*, not just the shape

**The shape check above cannot detect the failure that matters.** Every wrong pooling choice --
PAD token instead of last real token, mean instead of last, no instruction envelope, the envelope
applied to documents as well as queries -- still returns a `(1, 1024)` float32 vector and still
passes `hidden == 1024`. The check that catches them is a *relative* one: paraphrases must land
closer together than an unrelated string.

Save the block below as `check_embedder.py` and run `python check_embedder.py` (it is short enough
to paste into `python -` instead; it is not committed tooling):

```python
from axiom.config import get_settings
from axiom.indexing.embedder import load_embedder
import numpy as np

settings = get_settings()
emb = load_embedder(settings)
print("rung   :", emb.model_id, "dim", emb.dim)

a = "read a file from disk and return its contents as a string"
b = "load the contents of a file into memory and hand back the text"   # paraphrase of a
c = "compute the determinant of a square matrix by LU decomposition"   # unrelated

va, vb, vc = emb.encode([a, b, c])
ab, ac = float(va @ vb), float(va @ vc)
print(f"cos(a,b)={ab:.4f}  cos(a,c)={ac:.4f}  margin={ab - ac:+.4f}")

assert abs(float(np.linalg.norm(va)) - 1.0) < 1e-5, "vectors are not L2-normalised"
assert ab > ac + 0.05, "POOLING IS WRONG: paraphrases are not closer than an unrelated string"

# The query path must differ from the document path on an instruction-tuned model,
# and must NOT differ on a model that was never trained with the envelope.
doc_vec = emb.encode_one(a)
query_vec = emb.encode_query(a, settings.embedding_query_instruction)
differs = float(doc_vec @ query_vec) < 0.999
print("query envelope applied:", differs, "expected:", emb.needs_query_instruction)
assert differs == emb.needs_query_instruction, "query/document asymmetry is wrong for this rung"
print("OK")
```

Expected on the primary rung:

```
rung   : Qwen/Qwen3-Embedding-0.6B dim 1024
cos(a,b)=0.8xxx  cos(a,c)=0.4xxx  margin=+0.4xxx
query envelope applied: True expected: True
OK
```

The absolute cosines vary by model and are not the assertion; the **margin** and the
**query/document asymmetry** are. If the margin is near zero or negative, stop -- you are about
to index 8,765 documents into vectors that satisfy every other assertion in the codebase and
score like noise, which is indistinguishable from "the model just isn't good on APPS".

This runs against whichever rung actually loaded. On a box with no ONNX artifacts it exercises
the hash rung, where the margin is smaller but still positive -- that is the expected outcome,
not a failure, and the `rung   :` line tells you which case you are in. Wire this into the CI
smoke-index job ([_CONTRACT.md](_CONTRACT.md) §1), not only into this document.

---

## 7. Environment Variables

All settings are `pydantic-settings` v2 fields with the `AXIOM_` prefix, layered over the YAML profile named by `AXIOM_PROFILE`. Precedence, lowest to highest: `Settings` field default → profile YAML in `configs/` → `.env` → process environment → CLI flag. This is the same chain [Rules.md §7](Rules.md#7-configuration-discipline) states from the other direction, and it is the only statement of it.

**Every variable below is the field name in `src/axiom/config.py`, upper-cased under the prefix, with no renaming in between.** `AXIOM_DENSE_TOP_K` is `Settings.dense_top_k`; `AXIOM_AGENT_WALL_CLOCK_MS` is `Settings.agent_wall_clock_ms`. If another document names a field whose env var is not its own upper-case form, that document is wrong.

### 7.1 Core

| Variable | Type | Default | Effect |
|---|---|---|---|
| `AXIOM_PROFILE` | str | `default` | Loads `configs/<profile>.yaml`. Valid: `default`, `demo`, `eval`, `fast`, `accurate` — five, and `demo` is first-class, not a rename of `default` ([_CONTRACT.md](_CONTRACT.md) §3). |
| `AXIOM_INDEX_ROOT` | path | `.axiom` | Root of the on-disk index tree ([_CONTRACT.md](_CONTRACT.md) §6). |
| `AXIOM_DATA_ROOT` | path | `data` | **NOT IMPLEMENTED (2026-09-23)** — setting this has no effect. There is no `data_root` field on `Settings` and no code reads this name; `data/` is hard-coded at the call sites that use it. |
| `AXIOM_LOG_LEVEL` | str | `INFO` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR`. |
| `AXIOM_LOG_FORMAT` | str | `console` | `console` for humans, `json` for CI and the demo box. |
| `AXIOM_NUM_THREADS` | int | `0` | ONNX Runtime intra-op threads and FAISS `omp_set_num_threads`. `0` = physical core count. Set to 4 on a laptop to keep the UI responsive while indexing. |
| `AXIOM_SEED` | int | `42` | Seeds FAISS IVF training, the hash-embedder rung, and any sampling. Deterministic eval depends on it. `42` is the value `Settings.seed` carries; `1337` anywhere in this suite is stale. |
| `AXIOM_OFFLINE` | bool | `false` | Refuse all network access; also exports `HF_HUB_OFFLINE=1` to child processes. Use for the demo. |

### 7.2 Models

| Variable | Type | Default | Effect |
|---|---|---|---|
| `AXIOM_EMBEDDING_MODEL` | str | `Qwen/Qwen3-Embedding-0.6B` | HF repo id of the dense embedder. Set to `sentence-transformers/all-MiniLM-L6-v2` to use the fallback. |
| `AXIOM_EMBEDDING_BACKEND` | str | `onnx` | **NOT IMPLEMENTED (2026-09-23)** — setting this has no effect. There is no `embedding_backend` field and no torch runtime in `indexing/embedder.py`; the embedder is ONNX Runtime or a degraded rung, never torch. |
| `AXIOM_EMBEDDING_ONNX_PATH` | path | *(derived)* | **NOT IMPLEMENTED (2026-09-23)** — setting this has no effect. See the note below the table. The path is always derived from `embedding_model` by `_onnx_dir_for`; put the artifact where the derivation expects it (§7.2). |
| `AXIOM_EMBEDDING_DIM` | int | `1024` | Asserted against the loaded model at startup; a mismatch is fatal, not a warning. |
| `AXIOM_EMBEDDING_BATCH_SIZE` | int | `64` | Chunks per forward pass during indexing. Drop to `16` on an 8 GB box. |
| `AXIOM_EMBEDDING_MAX_TOKENS` | int | `512` | **NOT IMPLEMENTED (2026-09-23)** — setting this has no effect. See the note below the table. `512` is `DEFAULT_MAX_TOKENS` in `indexing/embedder.py` and cannot currently be changed from outside the code. **This matters:** `all-MiniLM-L6-v2` publishes `max_seq_length: 256`, so on the fallback rung Axiom truncates at twice the model's trained length. |
| `AXIOM_EMBEDDING_QUERY_INSTRUCTION` | str | `Given a natural-language question about a codebase, retrieve the code snippets that answer it` | The `{task}` half of the `Instruct: {task}\nQuery:{query}` envelope, applied on the **query side only**, and only by an instruction-tuned embedder (§6.2). Committed to `configs/` so a reported score is reproducible from a SHA. |
| `AXIOM_RERANKER_MODEL` | str | `BAAI/bge-reranker-v2-m3` | HF repo id of the cross-encoder. Profile-split: `eval.yaml` keeps this value, `demo.yaml` and `fast.yaml` set `cross-encoder/ms-marco-MiniLM-L-6-v2` ([_CONTRACT.md](_CONTRACT.md) §2). |
| `AXIOM_RERANK_MAX_CHARS` | int | `4096` | Per-document truncation window before cross-encoder scoring. Cost is quadratic in sequence length for the attention term, so this and `AXIOM_FUSION_TOP_N` are the two levers on rerank latency. `eval.yaml` sets `1024`. |
| `AXIOM_RERANKER_ENABLED` | bool | `true` | `false` returns the raw RRF ordering, and the response's `score_field` reads `rrf_score` rather than `rerank_score`. The accuracy cost is an ablation row we owe, not a figure this document may assert unmeasured. |
| `AXIOM_RERANKER_ONNX_PATH` | path | `data/models/onnx/bge-reranker-v2-m3-int8/model.onnx` | Overrides the derived artifact path. |
| `AXIOM_RERANKER_TIMEOUT_MS` | int | `2500` | Per-query rerank wall clock. On expiry the RRF order is returned and `RERANKER_TIMEOUT` is surfaced as a warning. |

> **Why some `AXIOM_*` overrides work and some silently do not.** There are two helpers named
> `_tunable` in the codebase and they do not behave the same way.
> `rerank/cross_encoder.py:207` resolves `Settings` field → `AXIOM_<FIELD>` environment variable →
> default, so every reranker tunable above (`AXIOM_RERANK_MAX_CHARS`, `AXIOM_RERANKER_ONNX_PATH`,
> `AXIOM_RERANK_LEXICAL_FALLBACK`, …) is a real knob even though `Settings` declares none of them.
> `indexing/embedder.py:82` is `return getattr(settings, field, default)` — it **never reads the
> environment**. So an embedder tunable that is not one of the 43 `Settings` fields is inert, and
> inert *silently*: you get the documented default and no warning.
>
> That is why `AXIOM_EMBEDDING_ONNX_PATH` and `AXIOM_EMBEDDING_MAX_TOKENS` are marked NOT
> IMPLEMENTED above while their reranker equivalents are not. The asymmetry is a defect, not a
> design: `Rules.md` §7 states one precedence chain for the whole project. Closing it is a
> three-line change to `indexing/embedder.py::_tunable` to match its sibling — but it changes
> runtime behaviour, so it is recorded here rather than done as part of a documentation pass.
>
> To check what is actually settable:
> `axiom --app-version` then `python -c "from axiom.config import Settings; print(sorted(Settings.model_fields))"`.
> Anything not in that list, and not read literally in `src/`, does nothing.

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
| `AXIOM_CHUNK_MIN_TARGET_TOKENS` | int | `64` | Lower end of the 64-512 target band. |
| `AXIOM_DENSE_ENABLED` | bool | `true` | `false` drops the dense signal entirely. |
| `AXIOM_SPARSE_ENABLED` | bool | `true` | `false` drops the sparse signal entirely. |
| `AXIOM_STRUCTURAL_ENABLED` | bool | `true` | `false` skips building and querying `structural.sqlite`. `eval.yaml` sets this, because APPS has no cross-file structure to index. |
| `AXIOM_EVAL_DENSE_WEIGHT` | float | `0.85` | Dense weight when the structural signal is off; the vector collapses to dense/sparse. |
| `AXIOM_EVAL_SPARSE_WEIGHT` | float | `0.15` | The sparse half. `# PLACEHOLDER`, `OQ-02`, swept on the **train** split by `T-112`. |

### 7.5 Versioning and evolutionary

| Variable | Type | Default | Effect |
|---|---|---|---|
| `AXIOM_ACTIVE_VERSION` | str \| null | `null` | **NOT IMPLEMENTED (2026-09-23)** — setting this has no effect. No `active_version` field exists on `Settings` and no code reads this name. Use `--version <id>` on the CLI, or `?version=` on the API, to pin a version. |
| `AXIOM_EVOLUTIONARY_ENABLED` | bool | `false` | Enables cross-version search and `SnippetFamily` grouping. Off by default because it widens candidate sets. |
| `AXIOM_DEDUPE_COSINE` | float | `0.95` | Family membership threshold within the same `symbol` + `file_path`. |
| `AXIOM_STABILITY_BONUS` | float | `0.10` | `final = base * (1 + bonus * stability)` for families in ≥2 versions. |

### 7.6 Services

| Variable | Type | Default | Effect |
|---|---|---|---|
| `AXIOM_API_HOST` | str | `127.0.0.1` | Bind address for `axiom serve`. Use `0.0.0.0` inside Docker. |
| `AXIOM_API_PORT` | int | `8000` | API port. |
| `AXIOM_API_CORS_ORIGINS` | str | *(unset)* | Comma-separated browser origins allowed to call the API, or `*`. Unset installs no CORS middleware at all. Needed only when a browser frontend is served from its own port — see [API.md §2](API.md#2-base-url-and-transport). |
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
# AXIOM_DATA_ROOT is NOT implemented - no code reads it; data/ is hard-coded
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
AXIOM_RERANK_MAX_CHARS=4096

# ---- agent ----
AXIOM_LLM_ENABLED=true
AXIOM_LLM_MODEL_PATH=data/models/gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf
AXIOM_AGENT_ENABLED=true
AXIOM_AGENT_MAX_PASSES=2
AXIOM_AGENT_WALL_CLOCK_MS=5000

# ---- retrieval ----
AXIOM_RRF_K=60
AXIOM_FAISS_IVF_THRESHOLD=50000
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
# Leave unset unless a browser frontend runs on its own port:
# AXIOM_API_CORS_ORIGINS=http://localhost:5173
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
axiom --app-version
axiom --help
```

Expected: `axiom 0.1.0`, then a usage block listing **ten** subcommands — `index`, `reindex`,
`query`, `classify`, `versions`, `families`, `eval`, `serve`, `ui`, `gc`. Eleven are *registered*;
`version` is `hidden=True`, so it works but is deliberately absent from `--help`. If you are
counting subcommands across documents: **nine** is the `FR-23` set (excludes `gc`), **ten** is what
`--help` prints, **eleven** is what `cli.py` registers. All three numbers are correct about
different things.

**`--app-version`, not `--version`.** The global `--version` flag takes a *string*: it names the
**index version id** to operate on, and `axiom --version` on its own is a usage error, not a
version banner. This trips people once and then never again.

There is no `axiom config show`. To see the resolved configuration:

```bash
python -c "from axiom.config import get_settings; import json; print(json.dumps(get_settings().model_dump(mode='json'), indent=2, default=str))"
```

The values to eyeball are `profile`, `index_root`, `embedding_model`, `embedding_dim`,
`reranker_model`, `reranker_enabled`, `llm_enabled`, `agent_max_passes`, `rrf_k`, `seed` (`42`).

`command not found` means you skipped `pip install -e .` or are in the wrong venv.

### Rung 3 — unit tests (~40 s)

```bash
pytest -q tests/
```

Expected tail: a `N passed` line with no failures, in well under a minute.

No `--timeout=120` — that flag needs `pytest-timeout`, which is not in the `dev` extra. Do not add
the flag without adding the plugin to `pyproject.toml` in the same commit, or the whole rung fails
with `unrecognized arguments` and looks like a broken install.

Skips are the model-dependent tests when no ONNX export or GGUF is present — acceptable. Any
failure here is a code problem, not a setup problem; see [TestPlan.md](TestPlan.md).

### Rung 4 — smoke index over the fixture repo (~25 s)

`tests/fixtures/repo_v1/` is a 13-file JavaScript toy repo committed specifically so setup can be
verified without the real corpus. It deliberately contains hostile cases — an unparseable file, a
circular import pair, an empty file, and two byte-identical duplicates — so this rung exercises the
degradation ladders rather than only the happy path. (`repo_v2/` is the same tree one "version"
later, and is what the reindex and evolutionary paths diff against.)

```bash
axiom index tests/fixtures/repo_v1 --version-id smoke --index-root /tmp/axiom-smoke
```

Expected — shape, not exact counts, which move whenever the chunker changes:

```
indexed tests/fixtures/repo_v1 as version smoke
  chunks     36 from 13 file(s)
  embedder   Qwen/Qwen3-Embedding-0.6B (dim 1024)
  dense      faiss / flat_ip
  sparse     bm25s
  structural built
  elapsed    276 ms
  degraded:  chunker: tree-sitter parse error, unsupported syntax -> regex identifier splitter
```

**That one `degraded:` line is expected and is the point** — `src/broken/syntax_error.js` is
unparseable on purpose, and the chunker is required to fall down its ladder rather than abort the
index. Any *other* degrade line is a real finding. In particular:

```
  embedder   axiom/hash-embedder-v1 (dim 1024)
  degraded:  indexing.embedder: ... no ONNX export at data/models/onnx/... -> next rung
```

means §6's export never ran, or the artifact is not where `_onnx_dir_for` derives its path
(`AXIOM_EMBEDDING_ONNX_PATH` will not redirect it — that variable is not implemented, §7.1). The index still
builds and still answers queries — the bottom rung is a seeded bag-of-token-hashes, not noise — but
it is **not** a semantic retriever, and no number measured on it means anything.

Then confirm the on-disk layout matches [_CONTRACT.md](_CONTRACT.md) §6:

```bash
find /tmp/axiom-smoke -maxdepth 3 | sort
# /tmp/axiom-smoke/registry.json
# /tmp/axiom-smoke/blobs/<content_hash>.npy          (one per distinct content)
# /tmp/axiom-smoke/index/smoke/manifest.json
# /tmp/axiom-smoke/index/smoke/chunks.jsonl
# /tmp/axiom-smoke/index/smoke/dense.faiss
# /tmp/axiom-smoke/index/smoke/dense.idmap.json
# /tmp/axiom-smoke/index/smoke/sparse.bm25s/          (bm25s native dir)
# /tmp/axiom-smoke/index/smoke/structural.sqlite
```

The blob count is lower than the chunk count on this fixture, and that is correct:
`src/dup/copy_a.js` and `copy_b.js` are byte-identical, so they share one `content_hash` and one
vector. That is `FR-19`'s dedupe working, visible on a 13-file repo.

### Rung 5 — smoke query returning ranked results (~2 s)

**The subcommand is `query`.** There is no `axiom search` — that name appears in older drafts and
produces `No such command 'search'`.

```bash
axiom query "how is user input normalized before dispatch" \
  --version smoke --index-root /tmp/axiom-smoke --top-k 3
```

Expected — a header, then one card per hit:

```
 query      'how is user input normalized before dispatch'
 type       SEMANTIC   passes 1 (sufficient)
 versions   smoke   profile default
 results    3 of top-k 3 in 267 ms

 1. src/utils/normalize.js:33-40  preprocessInput
    signals dense #1 · sparse #2
        why dense: function preprocessInput is a nearest neighbour of the query
            embedding (rank 1); corroborated at sparse rank 2
    33 export function preprocessInput(raw, options) {
    ...
                                                            score 0.0163
```

Real file paths, real line ranges, a per-signal breakdown and a `match_reason` on every card means
the full stack works: chunking, all three indices, RRF, hydration and the reranker.

**Read the `score` against the header, not against an absolute scale.** A score near `0.01` is an
RRF score (`1/(60+rank)`-scale) and means the cross-encoder did **not** run — it either was
disabled, timed out, or had no artifact to load, and the run degraded to passthrough. A score in
`[0, 1]` with meaningful spread is a calibrated rerank score. `axiom query --json` says which
outright, in `score_field`: `"rrf_score"` or `"rerank_score"`. Two orders of magnitude separate
them, and mistaking one for the other is how a demo accidentally overclaims.

Finally, prove the degraded path also works — this is the configuration the judges may hit:

```bash
AXIOM_LLM_ENABLED=false AXIOM_RERANKER_ENABLED=false \
  axiom query "normalize input" --version smoke --index-root /tmp/axiom-smoke --top-k 3
```

Same three hits, a `(heuristic)` classifier tag, and a lower latency. `NFR-07` requires this path to
be green at all times, not merely to exist.

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
| `No such command 'search'` | `axiom search` does not exist; the subcommand is `query` | `axiom query "<text>"`. Run `axiom --help` for the full list. |
| `axiom --version` reports a usage error instead of a version | The global `--version` names an **index version id**, not the app version | `axiom --app-version`. |
| `pytest: error: unrecognized arguments: --timeout=120` | `pytest-timeout` is not in the `dev` extra | Drop the flag (rung 3), or add `pytest-timeout` to `pyproject.toml`'s `dev` extra and re-sync. |
| Every query scores around `0.01` and `score_field` reads `rrf_score` | The cross-encoder never ran: no INT8 artifact, `AXIOM_RERANKER_ENABLED=false`, or a rerank timeout | Run §6.1's export, or raise `AXIOM_RERANKER_TIMEOUT_MS`. Results are still ranked — by RRF — but no accuracy number measured here is quotable. |
| Paraphrases are no closer than unrelated text, though `hidden` is 1024 | Wrong pooling or a missing query-side instruction envelope — the failure §6.4 exists to catch | Re-read §6.2, then re-run §6.4. Do not index anything until the margin is positive. |
| `axiom: command not found` after a successful install | `pip install -e .` skipped (pip fallback path), or the venv is not activated | Activate (`.venv\Scripts\activate` on Windows — §2.3), then `pip install -e .`. Confirm with `which axiom` / `where axiom`. |
| Streamlit UI loads but every query shows "API unreachable" | `axiom serve` not running, or `AXIOM_API_BASE_URL` points at the wrong host/port | Start the API first, then `curl -s localhost:8000/v1/health`. Inside Docker the UI must use the compose service name, not `127.0.0.1` — see [Deployment.md](Deployment.md) §3. |
| First query after startup takes 20-40 s, later ones are fast | Cold model load: ONNX session construction + GGUF mmap happen lazily on first use | Expected. Warm the process before the demo: `curl -s localhost:8000/v1/health?warm=true`. Never let the jury trigger the cold path — see the demo runbook in [Deployment.md](Deployment.md) §7. |
| `git diff` based reindex reports every file as modified | Repo cloned with `core.autocrlf=true` (Windows), so line endings differ from the indexed version | `git config core.autocrlf input` and re-clone, or reindex once to reset the baseline `file_hashes` in the manifest. |

---

## 10. Next Steps

| You want to | Go to |
|---|---|
| Understand the pipeline you just installed | [Design.md](Design.md) |
| Know the exact field names in results | [Schema.md](Schema.md) |
| Call the API or the CLI | [API.md](API.md) |
| Build a container, cut the release, or run the demo | [Deployment.md](Deployment.md) |
| Reproduce the NDCG@10 number | [TestPlan.md](TestPlan.md) for the protocol; [Tracker.md](Tracker.md) for the run log |
| Report something this doc did not cover | [OpenQuestions.md](OpenQuestions.md), as a new `OQ-##` |
