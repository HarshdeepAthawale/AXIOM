"""Sparse (BM25) index builder.

Writes ``sparse.bm25s/`` for one version, over the *identical* ``Chunk`` corpus
the dense builder consumed (Appflow.md flow 1, step 4). Corpus order is the
contract: bm25s document ``i`` corresponds to ``chunks.jsonl`` line ``i``, the
same correspondence FAISS row ``i`` carries (Schema.md section 15, invariant 12).
This module never re-chunks and never reorders.

Two backends, one artifact layout:

* **bm25s** (ADR-003, locked over ``rank-bm25`` on throughput at 10k-chunk
  scale) when the library imports and indexes cleanly.
* **A self-contained pure-Python BM25** otherwise, persisted as JSON.

The fallback is not defensive padding. On the ``eval`` profile the structural
signal is not built at all, so BM25 is one of only *two* signals carrying the
score (PRD.md section 2.2, TechSpecifications.md section 5.1.1); a sparse signal
that vanishes when an optional wheel is missing would take roughly a sixth of
the fused weight with it on the evaluator's machine. NFR-07 makes that
unacceptable, so the zero-dependency path is a first-class backend that produces
the same ``ScoredChunk`` lists, not an error message.

Both backends are driven with the same ``k1``/``b`` and the same Lucene-style
scoring family, so switching backends changes throughput, not ranking semantics.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from axiom.core.logging import get_logger, log_degradation
from axiom.retrieval.tokenizer import (
    DEFAULT_TOKENIZER,
    TOKENIZER_NAME,
    TOKENIZER_VERSION,
    CodeTokenizer,
    tokenize,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps import cost at zero
    from collections.abc import Iterable, Sequence

    from axiom.config import Settings
    from axiom.schema import Chunk

__all__ = [
    "BACKEND_BM25S",
    "BACKEND_PURE_PYTHON",
    "CHUNK_IDS_FILENAME",
    "FALLBACK_FILENAME",
    "META_FILENAME",
    "PORTABLE_FALLBACK_DEFAULT",
    "SPARSE_DIR_NAME",
    "SPARSE_SCHEMA_VERSION",
    "CodeTokenizer",
    "SparseIndexResult",
    "bm25_parameters",
    "build_index",
    "build_sparse_index",
    "chunk_token_stream",
    "sparse_dir",
    "tokenize",
]

_LOG = get_logger("indexing.sparse")

#: Directory name inside ``.axiom/index/<version_id>/``. Locked by
#: ``_CONTRACT.md`` section 6; renaming it is a breaking change.
SPARSE_DIR_NAME = "sparse.bm25s"

#: Tokeniser record (Schema.md section 14.6). bm25s does not persist its
#: tokeniser, so a query tokenised differently from the corpus silently returns
#: garbage -- this file is what makes that detectable instead of mysterious.
META_FILENAME = "prism_meta.json"

#: Positional ``document index -> chunk_id`` map. Deliberately the same shape as
#: ``dense.idmap.json`` (Schema.md section 14.4): an array, not an object keyed
#: by stringified integers, so the positional invariant is structural rather
#: than conventional. Additive file -- older readers that only know the bm25s
#: native format simply ignore it.
CHUNK_IDS_FILENAME = "chunk_ids.json"

#: The pure-Python backend's entire persisted state. Written whenever that
#: backend builds the index, and -- unless disabled -- alongside a successful
#: bm25s build too, so that an index built on a machine that has bm25s stays
#: queryable on one that does not. See :data:`PORTABLE_FALLBACK_DEFAULT`.
FALLBACK_FILENAME = "bm25_fallback.json"

#: On-disk schema generation for the JSON files this module writes.
SPARSE_SCHEMA_VERSION = 1

BACKEND_BM25S = "bm25s"
BACKEND_PURE_PYTHON = "pure_python"

#: Okapi BM25 term-saturation and length-normalisation constants.
#:
#: These are the one pair of algorithm constants in the sparse path that
#: TechSpecifications.md section 5 does not name, so there is no ``Settings``
#: field to read them from (Rules.md AP-07 lists the fields it does cover and
#: ``k1``/``b`` are not among them). They are named constants here rather than
#: literals at the callsite, and :func:`bm25_parameters` prefers a ``Settings``
#: field if one is ever added, so the config contract can absorb them without a
#: change at any callsite.
DEFAULT_BM25_K1 = 1.2
DEFAULT_BM25_B = 0.75

#: Whether a successful bm25s build also emits the dependency-free postings.
#:
#: On by default, and the reason is NFR-07 rather than caution. The bm25s native
#: format cannot be read without bm25s, and the read path may not rebuild an
#: index (Rules.md AP-12) -- so an index built on a developer machine that has
#: the wheel is simply *dead* on an evaluator machine that does not, silently
#: costing the sparse weight of every fused score. The sidecar is one extra
#: inversion pass at build time in exchange for that not being possible.
#: Overridable via a ``Settings.sparse_portable_fallback`` field if one is added.
PORTABLE_FALLBACK_DEFAULT = True


@dataclass(frozen=True, slots=True)
class SparseIndexResult:
    """What a build actually produced, for the manifest and the build log."""

    path: Path
    backend: str
    doc_count: int
    vocab_size: int
    token_count: int


def sparse_dir(index_dir: Path) -> Path:
    """Resolve the ``sparse.bm25s/`` directory for a version directory.

    Accepts either the version directory or the sparse directory itself. The
    tolerance is deliberate: "out_dir" is ambiguous between the two across the
    docs (Appflow.md says the builder *writes* ``sparse.bm25s/``, section 4.5's
    table says the same), and silently producing
    ``sparse.bm25s/sparse.bm25s/`` would only surface much later as an empty
    sparse signal.
    """
    path = Path(index_dir)
    return path if path.name == SPARSE_DIR_NAME else path / SPARSE_DIR_NAME


def bm25_parameters(settings: Settings | None = None) -> tuple[float, float]:
    """Return ``(k1, b)``, preferring ``Settings`` fields when they exist."""
    if settings is None:
        return DEFAULT_BM25_K1, DEFAULT_BM25_B
    k1 = float(getattr(settings, "sparse_bm25_k1", DEFAULT_BM25_K1))
    b = float(getattr(settings, "sparse_bm25_b", DEFAULT_BM25_B))
    return k1, b


def chunk_token_stream(chunk: Chunk, tokenizer: CodeTokenizer = DEFAULT_TOKENIZER) -> list[str]:
    """Tokens indexed for one chunk: its text, its symbol, and its callees.

    ``metadata.symbol`` and ``metadata.calls`` are folded in *on top of* the
    text they already appear in, which raises their term frequency rather than
    merely including them. That is the intended effect: the chunk that
    *declares* ``handleDeeplink`` should outrank a chunk that merely mentions it
    once, which is exactly what TC-043 asserts. Duplicates inside ``calls`` are
    preserved -- a function that calls ``validate`` three times really is more
    about validation than one that calls it once.

    Imports and the docstring are not folded in separately: the docstring is
    already inside ``text`` verbatim, and import specifiers are file-level facts
    that would attach the same tokens to every chunk in a file, flattening the
    IDF that makes BM25 discriminative in the first place.
    """
    tokens = tokenizer.tokenize(chunk.text)
    symbol = chunk.metadata.symbol
    if symbol:
        tokens.extend(tokenizer.tokenize(symbol))
    for callee in chunk.metadata.calls:
        tokens.extend(tokenizer.tokenize(callee))
    return tokens


def _atomic_write_json(path: Path, payload: object) -> None:
    """Write JSON via ``.tmp`` + :func:`os.replace`, sorted for byte stability.

    ``sort_keys`` matters beyond tidiness: an index artifact that differs byte
    for byte between two identical builds makes NFR-08's repeat-run diff
    unusable as evidence.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _build_postings(
    corpus_tokens: Sequence[Sequence[str]],
) -> tuple[dict[str, list[int]], list[int], int]:
    """Invert the corpus into ``term -> [doc, tf, doc, tf, ...]``.

    The postings list is flattened rather than a list of pairs purely for size:
    at ~10k chunks the nesting costs several megabytes of brackets for no
    information. Document indices are strictly ascending within a term, which is
    what lets the query path merge postings without sorting.
    """
    postings: dict[str, list[int]] = {}
    doc_lens: list[int] = []
    total = 0
    for doc_index, tokens in enumerate(corpus_tokens):
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        doc_lens.append(len(tokens))
        total += len(tokens)
        for term, freq in counts.items():
            postings.setdefault(term, []).extend((doc_index, freq))
    return postings, doc_lens, total


def _write_pure_python_index(
    target: Path,
    corpus_tokens: Sequence[Sequence[str]],
    k1: float,
    b: float,
) -> tuple[int, int]:
    """Persist the dependency-free BM25 backend. Returns ``(vocab_size, tokens)``."""
    postings, doc_lens, token_count = _build_postings(corpus_tokens)
    doc_count = len(doc_lens)
    payload = {
        "schema_version": SPARSE_SCHEMA_VERSION,
        "backend": BACKEND_PURE_PYTHON,
        "scoring": "lucene",
        "k1": k1,
        "b": b,
        "doc_count": doc_count,
        # avg_doc_len of an empty corpus is 0.0; the query path never divides by
        # it in that case because there are no postings to score.
        "avg_doc_len": (token_count / doc_count) if doc_count else 0.0,
        "doc_lens": doc_lens,
        "postings": postings,
    }
    _atomic_write_json(target / FALLBACK_FILENAME, payload)
    return len(postings), token_count


def _try_bm25s(
    target: Path,
    corpus_tokens: Sequence[Sequence[str]],
    k1: float,
    b: float,
) -> int | None:
    """Attempt the bm25s backend. Returns vocab size, or ``None`` if it degraded.

    The import lives here, not at module scope, because ``import axiom.indexing``
    must succeed on a bare install (NFR-07, Rules.md AP-06). The ``except
    Exception`` is intentionally broad rather than ``except ImportError``: an
    installed-but-incompatible bm25s that raises inside ``index()`` or ``save()``
    is exactly as fatal to the demo as a missing one, and the answer is the same
    rung of the ladder either way.
    """
    try:
        import bm25s

        retriever = bm25s.BM25(k1=k1, b=b, method="lucene")
        retriever.index(list(corpus_tokens))
        retriever.save(str(target))
    except ImportError as exc:
        log_degradation(_LOG, "indexing.sparse", f"bm25s unavailable ({exc})", "pure-python BM25")
        return None
    except Exception as exc:
        log_degradation(
            _LOG, "indexing.sparse", f"bm25s index/save failed ({exc!r})", "pure-python BM25"
        )
        return None

    vocab = getattr(retriever, "vocab_dict", None)
    return len(vocab) if vocab is not None else 0


def build_sparse_index(
    chunks: Iterable[Chunk],
    settings: Settings | None = None,
    out_dir: Path | str = Path("."),
) -> SparseIndexResult:
    """Build ``sparse.bm25s/`` over ``chunks`` and return what was written.

    Args:
        chunks: The version's chunk corpus, **in canonical order** -- sorted by
            ``(file_path, start_line)``, identical to ``chunks.jsonl`` line
            order. The order is not re-derived here; document index ``i`` simply
            *is* position ``i`` of this iterable.
        settings: Active settings. Only ``k1``/``b`` are read today; passing
            ``None`` uses the documented defaults.
        out_dir: The version directory (``.axiom/index/<version_id>/``), or the
            ``sparse.bm25s/`` directory itself. See :func:`sparse_dir`.

    Returns:
        A :class:`SparseIndexResult` naming the backend that actually ran, so
        the caller can record it in the build log and the manifest rather than
        inferring it from which files exist.

    An empty corpus is built, not refused: a zero-document index answers every
    query with an empty list, which fusion already handles, whereas raising here
    would abort a whole index run over an empty directory (Rules.md Rule 3).
    """
    target = sparse_dir(Path(out_dir))
    chunk_list = list(chunks)
    chunk_ids = [chunk.chunk_id for chunk in chunk_list]
    corpus_tokens = [chunk_token_stream(chunk) for chunk in chunk_list]
    token_count = sum(len(tokens) for tokens in corpus_tokens)
    k1, b = bm25_parameters(settings)

    # Write-once artifact: start from a clean directory so a rerun after a
    # half-finished build cannot leave one backend's files shadowing the other's.
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    backend = BACKEND_BM25S
    vocab_size = _try_bm25s(target, corpus_tokens, k1, b)
    if vocab_size is not None:
        if getattr(settings, "sparse_portable_fallback", PORTABLE_FALLBACK_DEFAULT):
            _write_pure_python_index(target, corpus_tokens, k1, b)
    else:
        # bm25s may have written partial state before failing; the pure-Python
        # backend must not inherit it.
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)
        backend = BACKEND_PURE_PYTHON
        vocab_size, _ = _write_pure_python_index(target, corpus_tokens, k1, b)

    _atomic_write_json(
        target / CHUNK_IDS_FILENAME,
        {
            "schema_version": SPARSE_SCHEMA_VERSION,
            "count": len(chunk_ids),
            "rows": chunk_ids,
        },
    )
    meta: dict[str, object] = {
        "schema_version": SPARSE_SCHEMA_VERSION,
        **DEFAULT_TOKENIZER.config(),
        "backend": backend,
        "k1": k1,
        "b": b,
        "doc_count": len(chunk_ids),
    }
    _atomic_write_json(target / META_FILENAME, meta)

    _LOG.info(
        "built sparse index: %d docs, %d terms, backend=%s",
        len(chunk_ids),
        vocab_size,
        backend,
        extra={
            "axiom_extra": {
                "stage": "sparse",
                "event": "index_built",
                "backend": backend,
                "doc_count": len(chunk_ids),
                "vocab_size": vocab_size,
                "token_count": token_count,
                "tokenizer_version": TOKENIZER_VERSION,
                "tokenizer": TOKENIZER_NAME,
                "path": str(target),
            }
        },
    )
    return SparseIndexResult(
        path=target,
        backend=backend,
        doc_count=len(chunk_ids),
        vocab_size=vocab_size,
        token_count=token_count,
    )


#: Appflow.md and TechSpecifications.md section 4.5 both name this entry point
#: ``indexing.sparse:build_index``, matching ``indexing.dense``/``structural``.
build_index = build_sparse_index
