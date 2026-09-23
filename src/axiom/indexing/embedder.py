"""Dense embedding and the content-addressed vector cache.

Three rungs, walked top-down by :func:`load_embedder`:

1. :class:`OnnxEmbedder` on the locked primary, ``Qwen/Qwen3-Embedding-0.6B``
   INT8 (dim 1024), via ONNX Runtime's CPU provider.
2. :class:`OnnxEmbedder` on the declared fallback,
   ``sentence-transformers/all-MiniLM-L6-v2`` (dim 384).
3. :class:`HashEmbedder` -- seeded, deterministic, and dependency-free.

The third rung exists because NFR-07 is judged on the evaluator's machine, not
ours: ``onnxruntime`` may be absent, the ``data/models/`` tree may never have
been downloaded, and the box may be offline. A pipeline that cannot embed is a
pipeline that returns nothing, so the bottom rung is a genuine bag-of-token-
hashes rather than noise -- it carries weak but real lexical signal, which keeps
the demo's dense leg honest instead of merely non-crashing.

Vectors are float32 and L2-normalised at the output boundary of every rung, so
FAISS inner product equals cosine everywhere downstream (Schema.md invariant 8,
Rules.md Rule 4). No rung is permitted to emit a distance.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from axiom.core.logging import get_logger, log_degradation

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from axiom.config import Settings
    from axiom.schema import Chunk

_LOG = get_logger("indexing.embedder")

#: Identity of the bottom rung. Deliberately not an HF id: ``VersionManifest``
#: pins a known dim for each real model id, and a synthetic id keeps the
#: hash embedder usable at whatever ``embedding_dim`` the profile asked for.
HASH_EMBEDDER_ID: Final = "axiom/hash-embedder-v1"

#: Declared fallback model and its dim, locked in _CONTRACT.md section 2.
FALLBACK_EMBEDDING_MODEL: Final = "sentence-transformers/all-MiniLM-L6-v2"
FALLBACK_EMBEDDING_DIM: Final = 384

#: Root of the locally-materialised INT8 ONNX exports (Setup.md section 7.2).
#: Axiom never downloads: a missing directory is a degrade, not a fetch.
ONNX_MODEL_ROOT: Final = Path("data/models/onnx")

#: Truncation length for a single sequence. Setup.md documents
#: ``AXIOM_EMBEDDING_MAX_TOKENS`` but ``Settings`` has no such field yet, so it
#: is read through :func:`_tunable` and this value is the documented default.
DEFAULT_MAX_TOKENS: Final = 512

#: Qwen3-Embedding is instruction-tuned and expects this envelope on the QUERY
#: side only; documents are encoded bare. Retrieval quality drops measurably if
#: the prefix is omitted, or if it is mistakenly applied to the corpus too.
QUERY_INSTRUCTION_TEMPLATE = "Instruct: {task}\nQuery:{query}"

#: Models whose training used that envelope. Matched as a substring of model_id.
INSTRUCTION_TUNED_MARKERS = ("qwen3-embedding",)

#: Signed-hash projections per token. Four keeps distinct identifiers close to
#: orthogonal at dim 384+ while costing one blake2b call per token.
_HASH_PROJECTIONS: Final = 4

#: Splits an identifier at camelCase humps, digit runs, and acronym/word seams.
_CAMEL_SPLIT = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[A-Za-z])(?=[0-9])")

#: Everything that is not an identifier character is a token boundary.
_WORD_SPLIT = re.compile(r"[^0-9A-Za-z_$]+")


def _tunable(settings: Settings, field: str, default: Any) -> Any:
    """Read a ``Settings`` field that the docs specify but the model may not carry.

    Setup.md section 7.2 and TestPlan.md TC-041 name several tunables
    (``embedding_max_tokens``, ``faiss_nprobe``) that the frozen ``Settings``
    does not yet declare. Reading them through here means the moment the field
    is added it takes effect, and until then the documented default applies from
    exactly one place rather than from a literal at a callsite (Rules.md AP-07).
    """
    return getattr(settings, field, default)


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise rows in place-safe fashion, mapping zero rows to a unit basis vector.

    A zero row would violate Schema.md invariant 8 (``abs(norm - 1) < 1e-5``) and
    would make FAISS inner product silently return 0.0 for that chunk rather
    than a similarity. Substituting ``e_0`` keeps the invariant true and keeps
    the degenerate chunk retrievable (it simply matches only other degenerate
    chunks), which is the whole spirit of Rule 3.
    """
    out = np.ascontiguousarray(matrix, dtype=np.float32)
    if out.size == 0:
        return out.reshape(out.shape)
    norms = np.linalg.norm(out, axis=1)
    degenerate = ~np.isfinite(norms) | (norms < 1e-12)
    if bool(degenerate.any()):
        out[degenerate] = 0.0
        out[degenerate, 0] = 1.0
        norms = np.linalg.norm(out, axis=1)
    out /= norms[:, None]
    return out


def tokenise_code(text: str) -> list[str]:
    """Code-aware tokenisation for the hash embedder.

    Mirrors the sparse tokeniser's contract (TechSpecifications.md section
    3.4.1): lowercase, split camelCase/snake_case/dots, and *keep the intact
    identifier too*, so ``resolveTool`` matches both ``resolve``/``tool`` and
    itself. No stopword list -- ``id``, ``in``, ``on`` are real code vocabulary.
    """
    tokens: list[str] = []
    for raw in _WORD_SPLIT.split(text):
        if not raw:
            continue
        intact = raw.lower()
        tokens.append(intact)
        parts = [p for chunk in raw.split("_") for p in _CAMEL_SPLIT.split(chunk) if p]
        if len(parts) > 1:
            tokens.extend(p.lower() for p in parts)
    return tokens


class Embedder(ABC):
    """Uniform text -> unit-vector interface shared by every rung of the ladder.

    Subclasses implement :meth:`_encode_batch`; the base class owns batching,
    normalisation, dtype, shape, and the call counters that TestPlan.md TC-031,
    TC-074 and TC-075 assert against ("embed call count == distinct content
    hashes", "a pure rename costs zero embedding calls").
    """

    def __init__(self, model_id: str, dim: int, batch_size: int) -> None:
        self.model_id = model_id
        self.dim = dim
        self.batch_size = max(1, batch_size)
        #: Number of texts actually pushed through the model. A cache hit never
        #: increments it, which is what makes the dedup tests observable.
        self.texts_encoded = 0
        #: Number of forward passes (batches) executed.
        self.batches_encoded = 0

    @abstractmethod
    def _encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Encode one batch, returning ``(len(texts), dim)`` un-normalised float32."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode texts into ``(n, dim)`` L2-normalised float32 vectors.

        Order is preserved exactly: row ``i`` is ``texts[i]``. This is load
        bearing -- ``chunks.jsonl`` line order, FAISS row order and idmap row
        order are the same order (Schema.md invariant 12).
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        blocks: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            raw = self._encode_batch(batch)
            self.texts_encoded += len(batch)
            self.batches_encoded += 1
            blocks.append(np.asarray(raw, dtype=np.float32).reshape(len(batch), self.dim))
        return _l2_normalise(np.vstack(blocks))

    def encode_one(self, text: str) -> np.ndarray:
        """Encode a single text into a ``(dim,)`` unit vector. Used on the query path."""
        vector: np.ndarray = self.encode([text])[0]
        return vector

    @property
    def needs_query_instruction(self) -> bool:
        """Whether this model expects an instruction envelope on query text.

        False for every rung except the instruction-tuned primary, which is why
        the decision lives on the embedder rather than at the call site: the
        ladder may have degraded to MiniLM or to the hash rung underneath us,
        and prefixing a query for a model that was not trained on the envelope
        makes retrieval worse, not better.
        """
        return any(marker in self.model_id.lower() for marker in INSTRUCTION_TUNED_MARKERS)

    def encode_query(self, text: str, task: str) -> np.ndarray:
        """Encode query text, applying the instruction envelope when required.

        This is the asymmetry that makes an instruction-tuned embedder work:
        queries are wrapped, documents are not. Index-time code paths must keep
        calling :meth:`encode`, never this -- wrapping the corpus would embed
        every chunk against the same prefix and collapse their separation.
        """
        payload = (
            QUERY_INSTRUCTION_TEMPLATE.format(task=task, query=text)
            if self.needs_query_instruction
            else text
        )
        return self.encode_one(payload)

    def close(self) -> None:  # noqa: B027 - deliberately optional, not abstract
        """Release any native handle.

        Idempotent and a no-op by default: only the ONNX rung owns a native
        handle, and forcing the pure-Python rung to implement a teardown it does
        not need would be ceremony, not safety.
        """

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"{type(self).__name__}(model_id={self.model_id!r}, dim={self.dim})"


class HashEmbedder(Embedder):
    """Deterministic signed-hash bag-of-tokens embedder. The bottom rung.

    Each code token is projected onto :data:`_HASH_PROJECTIONS` dimensions with
    a blake2b-derived index and sign, keyed by ``Settings.seed``; term counts are
    damped sublinearly (``1 + log(tf)``) exactly as BM25-family scorers do, so a
    long chunk that repeats one identifier does not swamp the vector.

    That makes it a real, if weak, lexical retriever: two texts sharing
    identifiers land close in inner product, and an exact-duplicate query of a
    chunk's text scores 1.0 against it (TestPlan.md TC-038 relies on precisely
    that identity property). It is not a semantic model and never claims to be
    -- it is the difference between a demo that ranks something sensible with
    zero models downloaded and a demo that returns an empty list.

    Determinism is total: no RNG, no wall clock, no dict-order dependence.
    """

    def __init__(self, dim: int, seed: int, batch_size: int = 64) -> None:
        super().__init__(model_id=HASH_EMBEDDER_ID, dim=max(1, dim), batch_size=batch_size)
        self._key = int(seed).to_bytes(8, "little", signed=False)

    def _project(self, token: str) -> list[tuple[int, float]]:
        """Map one token to (dimension, sign) pairs via a keyed blake2b digest."""
        digest = hashlib.blake2b(
            token.encode("utf-8"), digest_size=4 * _HASH_PROJECTIONS, key=self._key
        ).digest()
        pairs: list[tuple[int, float]] = []
        for index in range(_HASH_PROJECTIONS):
            word = int.from_bytes(digest[index * 4 : index * 4 + 4], "little")
            pairs.append(((word >> 1) % self.dim, 1.0 if word & 1 else -1.0))
        return pairs

    def _encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[str, int] = {}
            for token in tokenise_code(text):
                counts[token] = counts.get(token, 0) + 1
            if not counts:
                # All-punctuation or empty input: fall back to hashing the raw
                # string so the row is still a stable, non-degenerate vector.
                counts = {text.strip() or "\x00empty": 1}
            for token, count in counts.items():
                weight = 1.0 + math.log(count)
                for dimension, sign in self._project(token):
                    out[row, dimension] += np.float32(weight * sign)
        return out


class OnnxEmbedder(Embedder):
    """INT8 transformer encoder running on ONNX Runtime's CPU execution provider.

    Requires ``onnxruntime`` and ``tokenizers`` plus a locally materialised
    export under :data:`ONNX_MODEL_ROOT`. Nothing here downloads anything: an
    absent artefact is a degrade (Rules.md Rule 3), never a network fetch, which
    is what makes ``AXIOM_OFFLINE=true`` a no-op rather than a special case.

    The execution provider is pinned to CPU explicitly rather than left to
    auto-selection, per Rules.md section 6.3, and thread counts are pinned from
    ``Settings.num_threads`` because float reduction order moves scores in the
    4th decimal, which is enough to flip a tie (Rules.md section 5.2).
    """

    def __init__(
        self,
        model_id: str,
        dim: int,
        onnx_path: Path,
        tokenizer_path: Path,
        settings: Settings,
    ) -> None:
        super().__init__(model_id=model_id, dim=dim, batch_size=settings.embedding_batch_size)
        import onnxruntime as ort
        from tokenizers import Tokenizer

        options = ort.SessionOptions()
        if settings.num_threads > 0:
            options.intra_op_num_threads = settings.num_threads
            options.inter_op_num_threads = settings.num_threads
        self._session = ort.InferenceSession(
            str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self._session.get_inputs()}
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._max_tokens = int(_tunable(settings, "embedding_max_tokens", DEFAULT_MAX_TOKENS))
        self._tokenizer.enable_truncation(max_length=self._max_tokens)
        self._tokenizer.enable_padding(pad_id=self._pad_id(), pad_token=self._pad_token())
        #: Qwen3-Embedding pools the final token; BERT-family encoders mean-pool.
        self._last_token_pooling = "qwen3-embedding" in model_id.lower()

    def _pad_token(self) -> str:
        for candidate in ("<pad>", "[PAD]", "<|endoftext|>", "</s>"):
            if self._tokenizer.token_to_id(candidate) is not None:
                return candidate
        return "<pad>"

    def _pad_id(self) -> int:
        found = self._tokenizer.token_to_id(self._pad_token())
        return int(found) if found is not None else 0

    def _encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        encodings = self._tokenizer.encode_batch(list(texts))
        input_ids = np.asarray([e.ids for e in encodings], dtype=np.int64)
        mask = np.asarray([e.attention_mask for e in encodings], dtype=np.int64)
        feeds: dict[str, np.ndarray] = {}
        if "input_ids" in self._input_names:
            feeds["input_ids"] = input_ids
        if "attention_mask" in self._input_names:
            feeds["attention_mask"] = mask
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids)
        outputs = self._session.run(None, feeds)
        hidden = np.asarray(outputs[0], dtype=np.float32)
        if hidden.ndim == 2:
            # Export already pools (a sentence-embedding head); nothing to do.
            return hidden
        return self._pool(hidden, mask)

    def _pool(self, hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Reduce ``(batch, seq, dim)`` token states to ``(batch, dim)``."""
        if self._last_token_pooling:
            lengths = np.maximum(mask.sum(axis=1) - 1, 0)
            last: np.ndarray = hidden[np.arange(hidden.shape[0]), lengths]
            return last
        weights = mask.astype(np.float32)[:, :, None]
        summed = (hidden * weights).sum(axis=1)
        counts = np.maximum(weights.sum(axis=1), 1.0)
        pooled: np.ndarray = summed / counts
        return pooled

    def close(self) -> None:
        self._session = None

    @classmethod
    def load(cls, settings: Settings, model_id: str, dim: int) -> OnnxEmbedder:
        """Locate the INT8 export for ``model_id`` and open a session over it.

        Raises ``FileNotFoundError``/``ImportError`` on any missing piece so that
        :func:`load_embedder` can take the next rung. Callers other than the
        ladder should not use this directly.
        """
        directory = _onnx_dir_for(settings, model_id)
        override = _tunable(settings, "embedding_onnx_path", None)
        # The override names one artifact, so it can only mean the configured
        # model -- applying it to the fallback rung too would load the primary
        # export under the fallback's name and dim.
        if override and model_id == settings.embedding_model:
            onnx_path = Path(override)
        else:
            onnx_path = directory / "model.onnx"
        tokenizer_path = onnx_path.parent / "tokenizer.json"
        if not onnx_path.is_file():
            raise FileNotFoundError(f"no ONNX export at {onnx_path}")
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"no tokenizer.json beside {onnx_path}")
        return cls(model_id, dim, onnx_path, tokenizer_path, settings)


def _onnx_dir_for(settings: Settings, model_id: str) -> Path:
    """Derive the on-disk export directory for an HF model id.

    ``Qwen/Qwen3-Embedding-0.6B`` -> ``data/models/onnx/qwen3-embedding-0.6b-int8``,
    matching the paths Setup.md section 7.2 documents as defaults.
    """
    root = Path(_tunable(settings, "model_root", ONNX_MODEL_ROOT))
    return root / f"{model_id.rsplit('/', 1)[-1].lower()}-int8"


def load_embedder(settings: Settings) -> Embedder:
    """Walk the embedder ladder and return the best rung that actually loads.

    Never raises and never returns ``None``: the bottom rung has no dependencies
    beyond the standard library and numpy, so there is always an embedder. Every
    step down is logged through :func:`log_degradation`, because a silent
    fallback is indistinguishable from a silently wrong answer (Rules.md Rule 3).

    The caller must read :attr:`Embedder.model_id` and :attr:`Embedder.dim` off
    the returned object and record *those* in the manifest -- never the
    configured values, which may describe a rung that did not load.
    """
    primary = settings.embedding_model
    attempts: list[tuple[str, int]] = [(primary, settings.embedding_dim)]
    if primary != FALLBACK_EMBEDDING_MODEL:
        attempts.append((FALLBACK_EMBEDDING_MODEL, FALLBACK_EMBEDDING_DIM))

    for model_id, dim in attempts:
        try:
            embedder = OnnxEmbedder.load(settings, model_id, dim)
        except Exception as exc:
            log_degradation(
                _LOG,
                "indexing.embedder:load_embedder",
                f"{model_id} unavailable: {type(exc).__name__}: {exc}",
                "next rung of the embedder ladder",
            )
            continue
        _LOG.info(
            "embedder ready: %s (dim %d, onnxruntime CPU)",
            model_id,
            dim,
            extra={"axiom_extra": {"event": "embedder_loaded", "model": model_id, "dim": dim}},
        )
        return embedder

    log_degradation(
        _LOG,
        "indexing.embedder:load_embedder",
        "no ONNX embedder could be loaded",
        f"{HASH_EMBEDDER_ID} (seeded hashing, dim {settings.embedding_dim})",
    )
    return HashEmbedder(
        dim=settings.embedding_dim,
        seed=settings.seed,
        batch_size=settings.embedding_batch_size,
    )


class BlobCache:
    """Content-addressed vector store at ``.axiom/blobs/<content_hash>.npy``.

    Keyed by ``content_hash`` and nothing else -- not path, not version, not
    mtime (Rules.md AP-09). That single choice is what makes FR-19 true: a pure
    rename changes every ``chunk_id`` but no ``content_hash``, so it costs zero
    embedding calls (TestPlan.md TC-075), and a revert re-uses vectors computed
    two versions ago.

    Blobs are immutable: a file that exists is assumed correct and is never
    rewritten, which is what makes an interrupted build safely resumable
    (Schema.md invariant 14). The one thing verified on read is dimension -- a
    blob of the wrong dim belongs to a different model, and reusing it would mix
    384-dim and 1024-dim rows into one index. That case degrades to a re-embed
    with a loud warning rather than raising, so that a stale ``.axiom/blobs``
    left over from an earlier profile cannot brick a demo.
    """

    def __init__(self, root: Path, dim: int) -> None:
        self.root = Path(root)
        self.dim = dim
        self.hits = 0
        self.misses = 0

    @classmethod
    def for_settings(cls, settings: Settings, dim: int) -> BlobCache:
        """Open the cache under ``Settings.index_root`` (``.axiom/blobs`` by default)."""
        return cls(Path(settings.index_root) / "blobs", dim)

    def path_for(self, content_hash: str) -> Path:
        return self.root / f"{content_hash}.npy"

    def get(self, content_hash: str) -> np.ndarray | None:
        """Return the cached unit vector, or ``None`` on any miss or mismatch."""
        path = self.path_for(content_hash)
        if not path.is_file():
            self.misses += 1
            return None
        try:
            vector = np.load(path, allow_pickle=False)
        except Exception as exc:
            log_degradation(
                _LOG,
                "indexing.embedder:BlobCache.get",
                f"unreadable blob {path.name}: {type(exc).__name__}: {exc}",
                "re-embed this chunk",
            )
            self.misses += 1
            return None
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vector.shape != (self.dim,):
            log_degradation(
                _LOG,
                "indexing.embedder:BlobCache.get",
                f"blob {path.name} has dim {vector.shape[0]}, active embedder is dim {self.dim}",
                "re-embed this chunk; the stale blob is left untouched",
            )
            self.misses += 1
            return None
        self.hits += 1
        return vector

    def put(self, content_hash: str, vector: np.ndarray) -> None:
        """Write a blob once, atomically. An existing blob is never overwritten."""
        path = self.path_for(content_hash)
        if path.exists():
            return
        payload = np.ascontiguousarray(vector, dtype=np.float32).reshape(-1)
        if payload.shape != (self.dim,):
            raise ValueError(f"refusing to cache dim {payload.shape[0]} vector as dim {self.dim}")
        self.root.mkdir(parents=True, exist_ok=True)
        # os.replace is atomic within a filesystem, so a killed build leaves
        # either no blob or a complete one -- never a truncated array.
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        try:
            with tmp.open("wb") as handle:
                np.save(handle, payload, allow_pickle=False)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)


@dataclass(frozen=True)
class EmbeddingRun:
    """Outcome of embedding one chunk corpus: the matrix plus cache accounting."""

    vectors: np.ndarray
    """``(len(chunks), dim)`` float32, L2-normalised, in chunk order."""

    model_id: str
    """Identity of the embedder that actually ran. Goes into the manifest."""

    dim: int
    cache_hits: int
    embedded: int
    """Distinct contents pushed through the model. FR-19's headline number."""


def embed_chunks(
    chunks: Sequence[Chunk],
    embedder: Embedder,
    settings: Settings,
    cache: BlobCache | None = None,
) -> EmbeddingRun:
    """Embed a chunk corpus with content-addressed reuse, preserving chunk order.

    Dedup happens twice: against the on-disk blob cache (across versions and
    runs) and within this call (two chunks with identical normalised content
    embed once). TestPlan.md TC-031 asserts exactly this -- "embed call count ==
    distinct content hashes".

    ``cache=None`` opens the default ``.axiom/blobs`` store; pass an explicit
    cache to redirect it, and pass one rooted at a scratch path to disable
    sharing. There is no "no cache" mode because there is no case where
    re-embedding identical bytes is the right thing to do.
    """
    store = cache if cache is not None else BlobCache.for_settings(settings, embedder.dim)
    total = len(chunks)
    vectors = np.zeros((total, embedder.dim), dtype=np.float32)
    if total == 0:
        return EmbeddingRun(vectors, embedder.model_id, embedder.dim, store.hits, 0)

    pending_rows: dict[str, list[int]] = {}
    pending_text: dict[str, str] = {}
    hits = 0
    for row, chunk in enumerate(chunks):
        cached = store.get(chunk.content_hash)
        if cached is not None:
            vectors[row] = cached
            hits += 1
            continue
        if chunk.content_hash in pending_rows:
            pending_rows[chunk.content_hash].append(row)
            continue
        pending_rows[chunk.content_hash] = [row]
        pending_text[chunk.content_hash] = chunk.text

    # sorted() rather than dict order: iteration order reaches the batch
    # composition, and batch composition reaches float reduction order (NFR-08).
    order = sorted(pending_rows)
    embedded = 0
    batch_size = max(1, settings.embedding_batch_size)
    for start in range(0, len(order), batch_size):
        window = order[start : start + batch_size]
        encoded = embedder.encode([pending_text[h] for h in window])
        for offset, content_hash in enumerate(window):
            vector = encoded[offset]
            for row in pending_rows[content_hash]:
                vectors[row] = vector
            try:
                store.put(content_hash, vector)
            except Exception as exc:
                log_degradation(
                    _LOG,
                    "indexing.embedder:embed_chunks",
                    f"could not cache blob {content_hash}: {type(exc).__name__}: {exc}",
                    "in-memory vector only; the chunk re-embeds next build",
                )
        embedded += len(window)

    _LOG.info(
        "embedded %d/%d chunks (%d blob hits, %d distinct contents)",
        embedded,
        total,
        hits,
        len(order),
        extra={
            "axiom_extra": {
                "event": "embed_chunks",
                "chunks": total,
                "cache_hits": hits,
                "embedded": embedded,
                "model": embedder.model_id,
            }
        },
    )
    return EmbeddingRun(vectors, embedder.model_id, embedder.dim, hits, embedded)


__all__ = [
    "DEFAULT_MAX_TOKENS",
    "FALLBACK_EMBEDDING_DIM",
    "FALLBACK_EMBEDDING_MODEL",
    "HASH_EMBEDDER_ID",
    "ONNX_MODEL_ROOT",
    "BlobCache",
    "Embedder",
    "EmbeddingRun",
    "HashEmbedder",
    "OnnxEmbedder",
    "embed_chunks",
    "load_embedder",
    "tokenise_code",
]
