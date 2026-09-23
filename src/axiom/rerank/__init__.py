"""Stage 4 of the pipeline: cross-encoder reranking (FR-12).

Importing this package is free. Every heavy dependency (``onnxruntime``,
``transformers``, ``numpy``'s array machinery) is imported inside the function
that needs it, so ``axiom --help`` and ``pytest --collect-only`` never pay model
load cost (Rules.md AP-06, NFR-07).

Callers that only need the reordered list use :func:`rerank`. Callers that need
to know *which rung of the degradation ladder ran* -- above all
``agent/evaluator.py``, whose sufficiency predicate must switch from
``rerank_score`` to ``rrf_score`` when reranking degraded
(TechSpecifications.md section 5.3) -- use :func:`rerank_detailed` and read
:attr:`RerankOutcome.score_field`.
"""

from __future__ import annotations

from axiom.rerank.cross_encoder import (
    COMPONENT,
    DEFAULT_LEXICAL_K1,
    DEFAULT_LEXICAL_LOGIT_SPAN,
    DEFAULT_LEXICAL_PHRASE_BONUS,
    DEFAULT_RERANK_BATCH_SIZE,
    DEFAULT_RERANK_MAX_CHARS,
    DEFAULT_RERANK_MAX_TOKENS,
    FALLBACK_RERANKER_MODEL,
    STAGE,
    LexicalPairScorer,
    OnnxCrossEncoder,
    PairScorer,
    RerankMode,
    RerankOutcome,
    load_cross_encoder,
    rerank,
    rerank_detailed,
    sufficiency_score_field,
    truncate_document,
    warm_reranker,
)

__all__ = [
    "COMPONENT",
    "DEFAULT_LEXICAL_K1",
    "DEFAULT_LEXICAL_LOGIT_SPAN",
    "DEFAULT_LEXICAL_PHRASE_BONUS",
    "DEFAULT_RERANK_BATCH_SIZE",
    "DEFAULT_RERANK_MAX_CHARS",
    "DEFAULT_RERANK_MAX_TOKENS",
    "FALLBACK_RERANKER_MODEL",
    "STAGE",
    "LexicalPairScorer",
    "OnnxCrossEncoder",
    "PairScorer",
    "RerankMode",
    "RerankOutcome",
    "load_cross_encoder",
    "rerank",
    "rerank_detailed",
    "sufficiency_score_field",
    "truncate_document",
    "warm_reranker",
]
