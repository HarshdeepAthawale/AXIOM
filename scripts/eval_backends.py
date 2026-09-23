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

from collections.abc import Sequence
from dataclasses import dataclass, field

from axiom.config import Settings
from axiom.eval.mteb_adapter import EncoderBackend, LexicalBackend
from axiom.indexing.embedder import load_embedder
from axiom.schema import Chunk


def dense_only(settings: Settings) -> EncoderBackend:
    """Axiom's real embedder ladder, exact cosine, nothing else."""
    embedder = load_embedder(settings)
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
