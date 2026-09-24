"""Backend factories for ``scripts/run_eval.py --backend``.

Experiment drivers, not library code, which is why they live under ``scripts/``
next to ``run_eval.py`` rather than inside the package.

``dense_only`` is the ablation arm decision #3 asks for: the dense leg alone,
exact cosine, no sparse, no fusion, no rerank -- the honest baseline that the
unsourced "BGE 0.6B = 14.7" figure was standing in for.
``hybrid`` adds the BM25 leg at the eval-profile weights so the *gain* is
measurable against that same baseline rather than against a borrowed number.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from axiom.config import Settings
from axiom.core.hashing import compute_content_hash
from axiom.eval.mteb_adapter import EncoderBackend, LexicalBackend
from axiom.indexing.embedder import load_embedder
from axiom.schema import Chunk

# ---------------------------------------------------------------------------
# Embedding reuse across eval runs
# ---------------------------------------------------------------------------

#: Where cached eval vectors live. Deliberately NOT ``.axiom/blobs``.
#:
#: The production cache (``embedder.BlobCache``) is keyed by ``content_hash``
#: and nothing else, guarding only on dimension. That is exactly right for
#: FR-19 -- it is what makes a rename cost zero embeddings -- and it must not
#: change. But it also means two models that share a dimension share a key, and
#: the ablation arms here are precisely that case: ``dense_only`` and
#: ``dense_only_256`` are the same 384-dim model at two truncation lengths and
#: produce *different* vectors for the same text. Pointing the arms at the
#: production cache would silently serve 512-token vectors to the 256-token arm
#: and quietly invalidate the comparison the arm exists to make.
#:
#: So eval vectors get their own namespace, keyed by model identity and
#: truncation as well as content.
EVAL_CACHE_ROOT = Path("data/eval_cache")


def _namespace(encoder: object) -> str:
    """Directory name identifying everything that changes a vector."""
    model = str(getattr(encoder, "model_id", "unknown"))
    tokens = int(getattr(encoder, "_max_tokens", 0) or 0)
    dim = int(getattr(encoder, "dim", 0) or 0)
    slug = re.sub(r"[^A-Za-z0-9]+", "-", model).strip("-").lower()
    return f"{slug}-d{dim}-t{tokens}"


@dataclass
class CachedEncoder:
    """Wrap an embedder so identical text is embedded once per namespace.

    A full-split AppsRetrieval run embeds 8,765 documents and takes ~39 minutes
    with MiniLM on this box. Every ablation arm re-pays that cost for corpus
    text that did not change, which is what makes a four-arm ablation table
    expensive enough to skip. With this, only the first arm pays.

    Vectors are stored exactly as the encoder produced them, so this changes
    *when* work happens and never *what* the number is.
    """

    inner: Any
    root: Path = EVAL_CACHE_ROOT
    hits: int = 0
    misses: int = 0

    def __post_init__(self) -> None:
        self.dir = Path(self.root) / _namespace(self.inner)
        self.dir.mkdir(parents=True, exist_ok=True)

    @property
    def model_id(self) -> str:
        return str(getattr(self.inner, "model_id", "unknown"))

    @property
    def dim(self) -> int:
        return int(self.inner.dim)

    @property
    def degraded(self) -> bool:
        return bool(getattr(self.inner, "degraded", False))

    def _path(self, text: str) -> Path:
        return self.dir / f"{compute_content_hash(text)}.npy"

    def encode(self, texts: Sequence[str]) -> Any:
        import numpy as np

        texts = list(texts)
        out: list[Any] = [None] * len(texts)
        missing: list[int] = []
        for i, text in enumerate(texts):
            path = self._path(text)
            if path.is_file():
                try:
                    vector = np.load(path, allow_pickle=False)
                except Exception:
                    vector = None
                # A truncated or corrupt blob is a miss, never a wrong answer.
                if vector is not None and vector.shape == (self.dim,):
                    out[i] = vector
                    self.hits += 1
                    continue
            missing.append(i)

        if missing:
            fresh = np.asarray(self.inner.encode([texts[i] for i in missing]), dtype="float32")
            for slot, i in enumerate(missing):
                vector = fresh[slot]
                out[i] = vector
                self.misses += 1
                # Write-then-rename: an interrupted run must not leave a
                # half-written blob that a later run trusts. The handle is
                # explicit because np.save() appends ".npy" to any path that
                # does not already end in it, which silently defeats the
                # rename.
                final = self._path(texts[i])
                tmp = final.with_name(final.name + ".tmp")
                with tmp.open("wb") as handle:
                    np.save(handle, vector)
                tmp.replace(final)

        return np.vstack([np.asarray(v, dtype="float32") for v in out])


def _cached(encoder: Any, settings: Settings) -> Any:
    """Apply the eval cache unless AXIOM_EVAL_CACHE=0 disables it."""
    if os.environ.get("AXIOM_EVAL_CACHE", "1") in {"0", "false", "no"}:
        return encoder
    return CachedEncoder(inner=encoder)


def dense_only(settings: Settings) -> EncoderBackend:
    """Axiom's real embedder ladder, exact cosine, nothing else."""
    embedder = _cached(load_embedder(settings), settings)
    backend = EncoderBackend(encoder=embedder, batch_size=settings.embedding_batch_size)
    backend.name = f"dense-only:{embedder.model_id}"
    return backend


@dataclass
class HybridRrfBackend:
    """Dense + BM25 fused by reciprocal rank at the eval profile's weights."""

    dense: EncoderBackend
    sparse: LexicalBackend
    dense_weight: float
    sparse_weight: float
    rrf_k: int = 60
    name: str = "hybrid-rrf"
    _n: int = field(default=0, repr=False)

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._n = len(chunks)
        self.dense.index(chunks)
        self.sparse.index(chunks)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        depth = max(top_k, 100)
        fused: dict[str, float] = {}
        for hits, weight in (
            (self.dense.search(query, depth), self.dense_weight),
            (self.sparse.search(query, depth), self.sparse_weight),
        ):
            for rank, (chunk_id, _score) in enumerate(hits, start=1):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + weight / (self.rrf_k + rank)
        ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:top_k]

    @property
    def degraded(self) -> bool:
        return bool(getattr(self.dense, "degraded", False))

    @property
    def encoder(self) -> Any:
        """Surface the dense leg's embedder under the name provenance looks for.

        ``AxiomSearchModel.backend_detail`` reads ``backend.encoder`` to record
        which model actually answered, and ``run_eval.py`` compares that against
        the configured ``embedding_model`` -- the NFR-09 check that catches a run
        silently served by a fallback rung. Without this property the attribute
        is absent, the comparison is skipped, and a hybrid run whose embedder had
        degraded would still certify as reportable. A guard that cannot see the
        thing it guards is worse than no guard, because the artifact then claims
        a provenance nobody verified.
        """
        return getattr(self.dense, "encoder", None)


def hybrid(settings: Settings) -> HybridRrfBackend:
    return HybridRrfBackend(
        dense=dense_only(settings),
        sparse=LexicalBackend(),
        dense_weight=float(getattr(settings, "eval_dense_weight", 0.85)),
        sparse_weight=float(getattr(settings, "eval_sparse_weight", 0.15)),
    )


def sparse_only(settings: Settings) -> LexicalBackend:
    return LexicalBackend()


def dense_only_256(settings: Settings) -> EncoderBackend:
    """Same as :func:`dense_only` but truncating at the 256 tokens MiniLM was
    published with (``sentence_bert_config.json: max_seq_length = 256``), rather
    than the 512 in ``embedder.DEFAULT_MAX_TOKENS``. The ablation arm for
    whether that default costs retrieval quality on long documents.
    """
    backend = dense_only(settings)
    tokenizer = getattr(backend.encoder, "_tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("dense_only_256 requires the ONNX rung, not the hash rung")
    tokenizer.enable_truncation(max_length=256)
    backend.encoder._max_tokens = 256
    backend.name = f"dense-only-maxtok256:{backend.encoder.model_id}"
    return backend
