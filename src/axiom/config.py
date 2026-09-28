"""Settings and profile loading.

Precedence, per Rules.md section 7::

    CLI flag  >  env var (AXIOM_ prefix)  >  profile YAML  >  Settings field default

No magic numbers at a callsite: every algorithm constant in TechSpecifications.md
section 5 is a field here (Rules.md AP-07).

Constants marked PLACEHOLDER in the docs carry that status in
:data:`PLACEHOLDER_FIELDS`. A reported score built on an unresolved placeholder
is a Rules.md AP-14 violation, so the value is knowable programmatically rather
than only in prose.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from axiom.schema.enums import QueryType, SignalKind

#: Constants still carrying `PLACEHOLDER` status: chosen by judgement, not yet
#: measured. Rules.md AP-14 forbids a reported score from resting on one.
PLACEHOLDER_FIELDS: frozenset[str] = frozenset(
    {
        "chunk_min_tokens",
        "chunk_target_tokens",
        "agent_sufficiency_top1",
        "agent_sufficiency_floor",
        "stability_bonus",
    }
)

#: Placeholders that have since been measured, with the evidence. Kept as a
#: record rather than deleted: "this value was guessed and later confirmed" is a
#: different and stronger claim than "this value was always right", and the
#: distinction is exactly what AP-14 exists to preserve.
RESOLVED_PLACEHOLDERS: dict[str, str] = {
    "eval_sparse_weight": (
        "OQ-02 — swept 0.0-0.5 on the AppsRetrieval TRAIN partition (1,500 queries, "
        "5,000 docs); 0.15 maximises NDCG@10 at 26.02 against 25.06 for dense-only. "
        "data/sweeps/oq02_sparse_weight.json"
    ),
    "dedupe_cosine": (
        "OQ-11 — 45,753 chunk pairs over a 3-version corpus; 0.95 is the "
        "minimum-error threshold (1.0% members missed, 0.022% wrongly merged). "
        "data/sweeps/oq11_dedupe_cosine.json"
    ),
}

#: When a placeholder is *unreachable*, given the active settings, it cannot
#: affect the number, and flagging it against a run is a false positive that
#: teaches people to ignore the guard. A constant behind a disabled feature flag
#: is the clearest case: `configs/eval.yaml` turns the agent and the
#: evolutionary layer off, so their thresholds are never read during an eval.
#: Each entry answers "can this field be read under these settings?".
_REACHABLE_WHEN: dict[str, str] = {
    "agent_sufficiency_top1": "agent_enabled",
    "agent_sufficiency_floor": "agent_enabled",
    "stability_bonus": "evolutionary_enabled",
}

#: Default per-QueryType RRF weights, (dense, sparse, structural).
#: Locked in _CONTRACT.md section 5; every row sums to 1.0.
DEFAULT_STRATEGY_WEIGHTS: dict[QueryType, dict[SignalKind, float]] = {
    QueryType.SEMANTIC: {
        SignalKind.DENSE: 0.60,
        SignalKind.SPARSE: 0.30,
        SignalKind.STRUCTURAL: 0.10,
    },
    QueryType.STRUCTURAL: {
        SignalKind.DENSE: 0.20,
        SignalKind.SPARSE: 0.20,
        SignalKind.STRUCTURAL: 0.60,
    },
    QueryType.USAGE: {
        SignalKind.DENSE: 0.25,
        SignalKind.SPARSE: 0.55,
        SignalKind.STRUCTURAL: 0.20,
    },
    QueryType.HYBRID: {
        SignalKind.DENSE: 0.34,
        SignalKind.SPARSE: 0.33,
        SignalKind.STRUCTURAL: 0.33,
    },
}


class Settings(BaseSettings):
    """Every tunable in the system, in one place."""

    model_config = SettingsConfigDict(
        env_prefix="AXIOM_",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

    # --- Core -------------------------------------------------------------
    profile: str = Field(default="default", description="Active config profile name.")
    seed: int = Field(default=42, description="Global seed; FAISS IVF training uses it (NFR-08).")
    num_threads: int = Field(default=0, description="ONNX/torch thread cap. 0 means auto.")
    index_root: Path = Field(default=Path(".axiom"), description="On-disk index root.")
    log_level: str = Field(default="INFO")
    offline: bool = Field(default=False, description="Refuse network fetches; fail fast (RISK-11).")

    # --- Models -----------------------------------------------------------
    embedding_model: str = Field(default="Qwen/Qwen3-Embedding-0.6B")
    embedding_dim: int = Field(default=1024)
    embedding_batch_size: int = Field(default=64)
    embedding_query_instruction: str = Field(
        default="Given a natural-language question about a codebase, "
        "retrieve the code snippets that answer it",
        description="Task string for the instruction envelope an instruction-tuned "
        "embedder expects on the QUERY side only. Ignored by models that were not "
        "trained with one. Committed to configs/ so a reported score is reproducible.",
    )
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3")
    reranker_enabled: bool = Field(default=True)
    reranker_timeout_ms: int = Field(default=2500)
    rerank_max_chars: int = Field(
        default=4096,
        description="Per-document truncation before cross-encoder scoring. Cost is "
        "quadratic in sequence length for the attention term, so this is the main "
        "lever on rerank latency alongside fusion_top_n.",
    )
    llm_model: str = Field(default="Qwen2.5-1.5B-Instruct-Q4_K_M.gguf")
    llm_enabled: bool = Field(default=True)
    llm_max_tokens: int = Field(default=512)

    # --- Retrieval and fusion --------------------------------------------
    rrf_k: int = Field(default=60, description="RRF constant. Locked.")
    dense_top_k: int = Field(default=100)
    sparse_top_k: int = Field(default=100)
    structural_top_k: int = Field(default=50)
    fusion_top_n: int = Field(default=25, description="Post-fusion, pre-rerank width.")
    top_k_default: int = Field(default=10, description="Final result count.")
    faiss_ivf_threshold: int = Field(default=50_000)

    dense_enabled: bool = Field(default=True)
    sparse_enabled: bool = Field(default=True)
    structural_enabled: bool = Field(default=True)

    # --- Chunking ---------------------------------------------------------
    chunk_min_tokens: int = Field(default=16, description="PLACEHOLDER, OQ-09.")
    chunk_target_tokens: int = Field(default=512, description="PLACEHOLDER, OQ-09.")
    chunk_min_target_tokens: int = Field(default=64, description="PLACEHOLDER, OQ-09.")

    # --- Agent ------------------------------------------------------------
    agent_enabled: bool = Field(default=True)
    agent_max_passes: int = Field(default=2, description="Locked.")
    agent_wall_clock_ms: int = Field(default=5000, description="Locked.")
    agent_sufficiency_top1: float = Field(default=0.35, description="PLACEHOLDER, OQ-10.")
    agent_sufficiency_floor: float = Field(default=0.20, description="PLACEHOLDER, OQ-10.")
    agent_sufficiency_min_results: int = Field(default=3, description="Locked.")

    # --- Versioning and evolutionary -------------------------------------
    evolutionary_enabled: bool = Field(default=False)
    dedupe_cosine: float = Field(
        default=0.95,
        description="Measured: OQ-11, minimum-error threshold over 45,753 pairs.",
    )
    stability_bonus: float = Field(default=0.10, description="PLACEHOLDER, OQ-11.")

    # --- Eval -------------------------------------------------------------
    eval_sparse_weight: float = Field(
        default=0.15,
        description="Measured: OQ-02, swept on the train partition; peak NDCG@10.",
    )
    eval_dense_weight: float = Field(default=0.85)

    # --- Services ---------------------------------------------------------
    api_port: int = Field(default=8000)
    ui_port: int = Field(default=8501)

    def strategy_weights_for(self, query_type: QueryType) -> dict[SignalKind, float]:
        """Nominal weight vector for a query type, before empty-signal renormalisation.

        On the ``eval`` profile the structural signal is not merely down-weighted,
        it is not built at all (PRD.md section 2.2), so the vector collapses to
        dense/sparse at the tuned eval weights.
        """
        if not self.structural_enabled:
            total = self.eval_dense_weight + self.eval_sparse_weight
            return {
                SignalKind.DENSE: self.eval_dense_weight / total,
                SignalKind.SPARSE: self.eval_sparse_weight / total,
            }
        return dict(DEFAULT_STRATEGY_WEIGHTS[query_type])

    def active_placeholders(self) -> list[str]:
        """Unmeasured constants that this configuration can actually read.

        The AP-14 guard on reported scores consumes this. It is deliberately
        narrower than :data:`PLACEHOLDER_FIELDS`: a threshold behind a disabled
        feature flag cannot influence the run, and reporting it would be a false
        positive. Anything whose gate is *on* is reported, as is anything with
        no gate at all.
        """
        active: list[str] = []
        for field in PLACEHOLDER_FIELDS:
            gate = _REACHABLE_WHEN.get(field)
            if gate is None or bool(getattr(self, gate, True)):
                active.append(field)
        return sorted(active)

    def all_placeholders(self) -> list[str]:
        """Every field still carrying placeholder status, gated or not."""
        return sorted(PLACEHOLDER_FIELDS)


def load_profile(name: str, configs_dir: Path | None = None) -> dict[str, Any]:
    """Read ``configs/<name>.yaml`` and return it as a flat override mapping."""
    root = configs_dir or Path("configs")
    path = root / f"{name}.yaml"
    if not path.is_file():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"profile {path} must contain a mapping, got {type(loaded).__name__}")
    flat: dict[str, Any] = {}
    for key, value in loaded.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat[f"{key}_{sub_key}"] = sub_value
        else:
            flat[key] = value
    return flat


def get_settings(
    profile: str | None = None,
    configs_dir: Path | None = None,
    **overrides: Any,
) -> Settings:
    """Build :class:`Settings` honouring the full precedence chain.

    ``overrides`` stands in for CLI flags and wins over everything. Env vars beat
    the profile YAML, which beats the field defaults.

    Both the profile mapping and the overrides arrive as init kwargs, and
    pydantic-settings ranks init kwargs *above* the environment. So any profile
    key that the environment also sets is dropped here before construction --
    otherwise a YAML value would silently outrank ``AXIOM_*``, inverting the
    documented order.
    """
    name = profile or Settings().profile
    base = load_profile(name, configs_dir)
    base["profile"] = name

    env_set = {field for field in Settings.model_fields if f"AXIOM_{field.upper()}" in os.environ}
    base = {k: v for k, v in base.items() if k not in env_set}

    base.update({k: v for k, v in overrides.items() if v is not None})
    return Settings(**base)


#: Default location of the exported models, relative to the working directory.
MODEL_DIR = Path("data/models")


def resolve_model_dir() -> Path:
    """Where the exported models live: the directory holding ``onnx/`` and ``gguf/``.

    ``AXIOM_MODEL_DIR`` wins when set. Otherwise ``./data/models`` if it exists,
    else ``data/models`` in the Axiom checkout this package runs from. The last
    rung matters because ``axiom reindex`` has to run inside the repository being
    indexed, where a working-directory-relative path finds no models and every
    rung of the ladder quietly degrades to the hashing embedder.
    """
    configured = os.environ.get("AXIOM_MODEL_DIR")
    if configured:
        return Path(configured)
    if MODEL_DIR.is_dir():
        return MODEL_DIR
    checkout = Path(__file__).resolve().parents[2] / MODEL_DIR
    return checkout if checkout.is_dir() else MODEL_DIR


def onnx_model_file(directory: Path) -> Path:
    """The ONNX graph in an export directory.

    ``model.onnx`` is the documented name, but ``optimum-cli onnxruntime quantize``
    writes ``model_quantized.onnx``, and following Setup.md section 6.1 to the
    letter left an export Axiom could not see. Accept either, preferring the
    documented name.
    """
    documented = directory / "model.onnx"
    quantized = directory / "model_quantized.onnx"
    if not documented.is_file() and quantized.is_file():
        return quantized
    return documented
