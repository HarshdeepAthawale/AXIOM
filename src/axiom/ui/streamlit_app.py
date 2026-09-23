"""The jury-facing demo UI (FR-25, US-11).

US-11 states the job precisely: *"I want to click through a query and see signal
contributions, so that I can tell the hybrid architecture is real and not a
slide."* Everything here is arranged around that sentence. The per-signal
breakdown is the centrepiece of every result card -- the rank each of dense,
sparse and structural gave this chunk, the weighted RRF term each contributed,
and which one dominated -- because that panel is the only thing on screen that a
single-embedding-model demo could not fake.

Three design decisions worth stating, because each one had a tempting wrong
answer:

**The numbers are reconstructed, not invented.** ``RetrievalResult`` carries the
per-signal ranks (``signals``) but not the fused arithmetic, and the pipeline's
response does not ship ``FusedResult``. Rather than display a plausible-looking
bar of unknown provenance, the card recomputes each term as
``w_i / (rrf_k + rank_i)`` from the plan's own (already renormalised) weights and
``Settings.rrf_k`` -- the identical formula fusion used, over the identical
inputs, so the panel is a derivation rather than a decoration. When the
reconstruction cannot be trusted (a weight missing from the plan) the card says
so instead of drawing a bar.

**The score is labelled with the field it came from.** With no cross-encoder
installed, ``score`` is a raw RRF value of order 0.016, which reads as "no
confidence" to anyone who assumes a probability. The response carries
``score_field``; the card prints it. Rescaling RRF into a pretty ``[0, 1]`` would
be an uncalibrated invention and the demo would be lying in a way nobody could
catch from the screen.

**Retrieval runs in-process when the HTTP API is not up.** Deployment.md's
compose file points the UI at ``http://api:8000``; a laptop running only
``streamlit run`` has nothing on 8000. The UI probes, uses the API when it
answers, and otherwise calls the same pipeline the API would have called --
through :mod:`axiom.api.routes`, so both paths share one implementation and one
error mapping. The sidebar always says which one is live.

Streamlit is an optional dependency: every Streamlit symbol is imported inside
:func:`main`, so this module imports on a bare install and the helpers below are
testable with nothing installed at all.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from axiom.api.models import (
    ErrorCode,
    QueryRequest,
    QueryResponse,
    VersionsResponse,
)
from axiom.api.routes import ApiError
from axiom.config import Settings, get_settings
from axiom.core.logging import get_logger, log_degradation
from axiom.schema import Chunk, QueryPlan, QueryType, RetrievalResult, SignalKind, SnippetFamily
from axiom.ui import DEFAULT_API_BASE_URL

_LOG = get_logger("ui.app")

#: Seconds to wait for the API health probe. Short on purpose: this runs on the
#: first page load and a refused connection is the normal, uninteresting case.
PROBE_TIMEOUT_S = 1.0

#: Seconds to wait for a query over HTTP. Generously above the agent loop's own
#: 5 s budget so a slow-but-working answer is never cut off by the client.
QUERY_TIMEOUT_S = 120.0

#: Display order and colour for the three signals. Fixed so a card's bars always
#: appear in the same order, whatever order the mapping happens to iterate.
SIGNAL_ORDER: tuple[SignalKind, ...] = (
    SignalKind.DENSE,
    SignalKind.SPARSE,
    SignalKind.STRUCTURAL,
)

#: Hues chosen to stay distinguishable on both the light and the dark Streamlit
#: theme, and to survive the most common colour-vision deficiency (they differ in
#: lightness as well as hue, so the bars are not distinguished by colour alone --
#: each one is labelled with its signal name and rank).
SIGNAL_COLOUR: dict[SignalKind, str] = {
    SignalKind.DENSE: "#4C78A8",
    SignalKind.SPARSE: "#E1A100",
    SignalKind.STRUCTURAL: "#3F9E6C",
}

#: One-word gloss per signal, for the card legend.
SIGNAL_GLOSS: dict[SignalKind, str] = {
    SignalKind.DENSE: "embedding similarity",
    SignalKind.SPARSE: "BM25 term overlap",
    SignalKind.STRUCTURAL: "call/import graph",
}

#: Badge colour per query type, so the classification is readable at a glance.
QUERY_TYPE_COLOUR: dict[QueryType, str] = {
    QueryType.SEMANTIC: "#4C78A8",
    QueryType.STRUCTURAL: "#3F9E6C",
    QueryType.USAGE: "#E1A100",
    QueryType.HYBRID: "#8E6FBF",
}

#: File extension -> the language name Streamlit's highlighter understands.
LANGUAGE_BY_SUFFIX: dict[str, str] = {
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".py": "python",
    ".json": "json",
    ".md": "markdown",
    ".html": "html",
    ".css": "css",
    ".sql": "sql",
    ".sh": "bash",
}

#: Backend selection modes offered in the sidebar.
BACKEND_AUTO = "auto"
BACKEND_HTTP = "http"
BACKEND_LOCAL = "local"

#: Remediation shown when no index resolves -- TC-072 asks for an actionable
#: empty state here, not a traceback.
NO_INDEX_HINT = (
    "No index found. Build one first:\n\n```bash\naxiom index /path/to/repo --version-id v1\n```"
)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackendChoice:
    """Which retrieval path is live, and why.

    ``notes`` exists so the sidebar can explain a fallback instead of silently
    changing behaviour -- the same discipline the pipeline applies to its own
    degradation ladders (NFR-07).
    """

    kind: str
    label: str
    base_url: str | None = None
    notes: list[str] = field(default_factory=list)


class LocalBackend:
    """Retrieval in this process, through the same handlers the API exposes.

    Calling :mod:`axiom.api.routes` rather than :mod:`axiom.pipeline` directly is
    the point: the UI then shares the API's version resolution, its error
    mapping and its envelope, so "the UI shows something different from the API"
    is not a bug that can exist.
    """

    kind = BACKEND_LOCAL

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def label(self) -> str:
        return "in-process pipeline"

    def query(self, request: QueryRequest) -> QueryResponse:
        from axiom.api.routes import run_query

        return run_query(request, self.settings)

    def versions(self) -> VersionsResponse:
        from axiom.api.routes import list_versions

        return list_versions(self.settings)


class HttpBackend:
    """Retrieval over ``POST /v1/query`` on a running ``axiom serve``.

    Uses :mod:`urllib` rather than ``requests`` or ``httpx``: neither is a
    declared dependency, and adding one so a demo page can make two GETs would
    put a new install step between an evaluator and a working UI.
    """

    kind = BACKEND_HTTP

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    @property
    def label(self) -> str:
        return f"HTTP API at {self.base_url}"

    def query(self, request: QueryRequest) -> QueryResponse:
        payload = request.model_dump(mode="json", exclude_none=True)
        body = _http_json(f"{self.base_url}/v1/query", payload, QUERY_TIMEOUT_S)
        return QueryResponse.model_validate(body)

    def versions(self) -> VersionsResponse:
        body = _http_json(f"{self.base_url}/v1/versions", None, PROBE_TIMEOUT_S * 5)
        return VersionsResponse.model_validate(body)


def _http_json(url: str, payload: dict[str, Any] | None, timeout: float) -> Any:
    """One JSON request, with the API's error body translated into :class:`ApiError`.

    The server already speaks one error shape (API.md section 6); decoding it
    back into the same exception the local backend raises means the rendering
    code never has to ask which backend produced a failure.
    """
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST" if data is not None else "GET",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        code = ErrorCode.INTERNAL_ERROR
        message = detail.strip() or f"HTTP {exc.code}"
        try:
            parsed = json.loads(detail)
            code = ErrorCode(parsed.get("error", ErrorCode.INTERNAL_ERROR.value))
            message = str(parsed.get("detail", message))
        except (ValueError, KeyError):
            pass
        raise ApiError(exc.code, code, message) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise ApiError(503, ErrorCode.INDEX_UNAVAILABLE, f"{url} unreachable: {exc}") from exc


def api_is_up(base_url: str, timeout: float = PROBE_TIMEOUT_S) -> bool:
    """Whether ``GET /v1/health`` answers. Never raises."""
    try:
        body = _http_json(f"{base_url.rstrip('/')}/v1/health", None, timeout)
    except Exception:
        return False
    return isinstance(body, dict) and body.get("status") == "ok"


def choose_backend(
    mode: str, base_url: str, settings: Settings
) -> tuple[LocalBackend | HttpBackend, BackendChoice]:
    """Resolve the retrieval path for this session.

    ``auto`` prefers the HTTP API when it answers -- that is the deployed
    topology Deployment.md describes, and exercising it in the demo proves the
    API is real -- and falls back to in-process retrieval when nothing is
    listening, with the fallback stated in the sidebar rather than hidden.
    """
    notes: list[str] = []
    if mode == BACKEND_LOCAL:
        pinned = LocalBackend(settings)
        return pinned, BackendChoice(BACKEND_LOCAL, pinned.label, None, notes)

    if mode == BACKEND_HTTP:
        forced = HttpBackend(base_url)
        return forced, BackendChoice(BACKEND_HTTP, forced.label, base_url, notes)

    if api_is_up(base_url):
        http = HttpBackend(base_url)
        return http, BackendChoice(BACKEND_HTTP, http.label, base_url, notes)

    log_degradation(
        _LOG,
        "ui:choose_backend",
        f"no API answering at {base_url}",
        "in-process pipeline",
    )
    notes.append(f"No API answering at {base_url}; queries run in this process instead.")
    local = LocalBackend(settings)
    return local, BackendChoice(BACKEND_LOCAL, local.label, base_url, notes)


# ---------------------------------------------------------------------------
# Pure helpers -- no Streamlit, so they are testable with nothing installed
# ---------------------------------------------------------------------------


def language_for(file_path: str) -> str:
    """Highlighter language for a path, defaulting to unhighlighted text.

    Guessing wrong is worse than not guessing: a mis-highlighted snippet reads as
    broken code on a projector.
    """
    return LANGUAGE_BY_SUFFIX.get(PurePosixPath(file_path).suffix.lower(), "text")


def rrf_terms(
    signals: dict[SignalKind, int], weights: dict[SignalKind, float], rrf_k: int
) -> dict[SignalKind, float]:
    """Per-signal RRF terms ``w_i / (rrf_k + rank_i)`` for one result.

    The identical arithmetic ``retrieval.fusion`` performed, over the identical
    inputs: the ranks come from ``RetrievalResult.signals`` (copied verbatim from
    ``FusedResult.contributions``) and the weights from the returned
    ``QueryPlan``, which carries the *renormalised* vector of the winning pass.
    A signal the plan gives no weight contributes nothing and is omitted, exactly
    as fusion omitted it.
    """
    terms: dict[SignalKind, float] = {}
    for signal, rank in signals.items():
        weight = weights.get(signal)
        if weight is None or weight <= 0.0 or rank < 1:
            continue
        terms[signal] = weight / (rrf_k + rank)
    return terms


def reconstructed_rrf(result: RetrievalResult, plan: QueryPlan, rrf_k: int) -> float | None:
    """The result's fused score, or ``None`` when it cannot be reconstructed.

    ``None`` happens when the plan carries no positive weight for any signal that
    surfaced this chunk -- possible if a caller passed a hand-built plan. The
    card then omits the breakdown rather than drawing a bar of zeroes that would
    look like a genuine "no contribution" reading.
    """
    terms = rrf_terms(result.signals, plan.strategy_weights, rrf_k)
    if not terms:
        return None
    return sum(terms.values())


def dominant_signal(terms: dict[SignalKind, float]) -> SignalKind | None:
    """Largest-term signal, ties broken DENSE, SPARSE, STRUCTURAL (Schema.md section 8)."""
    best: SignalKind | None = None
    best_term = float("-inf")
    for signal in SIGNAL_ORDER:
        term = terms.get(signal)
        if term is not None and term > best_term:
            best, best_term = signal, term
    return best


def rerank_delta(result: RetrievalResult, plan: QueryPlan, rrf_k: int) -> float | None:
    """``final score - first-stage RRF score``, or ``None`` when not reconstructable.

    On the passthrough rung this is exactly zero and the card says "reranker
    passthrough" rather than showing a meaningless 0.0000: the reranker did not
    move anything because the reranker did not run.
    """
    base = reconstructed_rrf(result, plan, rrf_k)
    if base is None:
        return None
    return result.score - base


def rank_movement(response: QueryResponse, rrf_k: int) -> dict[str, int]:
    """``chunk_id -> positions gained`` between the first-stage and final order.

    Positive means the second stage promoted the chunk. Computed only over the
    returned window, so it answers "how did the reranker reorder what you are
    looking at", which is the question the screen actually poses.
    """
    scored = [
        (result.chunk.chunk_id, reconstructed_rrf(result, response.query_plan, rrf_k))
        for result in response.results
    ]
    if any(value is None for _, value in scored):
        return {}
    ordered = sorted(scored, key=lambda item: (-(item[1] or 0.0), item[0]))
    first_stage = {chunk_id: position for position, (chunk_id, _) in enumerate(ordered, start=1)}
    return {
        result.chunk.chunk_id: first_stage[result.chunk.chunk_id] - position
        for position, result in enumerate(response.results, start=1)
    }


def snippet_families(
    results: list[RetrievalResult], settings: Settings, version_ids: list[str]
) -> tuple[dict[str, SnippetFamily], list[str]]:
    """Group the displayed results into snippet families across versions (FR-21).

    Reads the local index directly rather than asking the API, because there is
    no families endpoint (API.md section 3 defines four) and because the UI and
    the index sit on the same machine in every topology this project ships --
    including Deployment.md's compose file, where both containers mount the same
    volume.

    Only the ``(symbol, file_path)`` keys of the results on screen are loaded.
    Grouping requires matching key *and* embedding similarity (TC-082), so a
    version whose blobs are missing yields identity families, and the expander
    says so rather than implying the comparison happened.

    Returns:
        ``(family_by_chunk_id, notes)``. Both empty is the ordinary answer for a
        single-version index.
    """
    from axiom.indexing import manifest as mf
    from axiom.versioning.evolutionary import build_families

    notes: list[str] = []
    keys = {(r.chunk.metadata.symbol, r.chunk.location.file_path) for r in results}
    if not keys or not version_ids:
        return {}, notes

    by_version: dict[str, list[Chunk]] = {}
    embeddings: dict[str, list[float]] = {}
    missing_blobs = 0
    for version_id in version_ids:
        members: list[Chunk] = []
        manifest = mf.try_read_manifest(settings, version_id)
        dim = manifest.embedding_dim if manifest is not None else None
        for record in mf.iter_chunk_records(mf.chunks_path(settings, version_id)):
            location = record.get("location") or {}
            metadata = record.get("metadata") or {}
            if (metadata.get("symbol"), location.get("file_path")) not in keys:
                continue
            try:
                chunk = Chunk.model_validate(record)
            except Exception:
                continue
            members.append(chunk)
            if chunk.content_hash in embeddings:
                continue
            try:
                vector = mf.load_blob(settings, chunk.content_hash, expected_dim=dim)
            except Exception:
                vector = None
            if vector is None:
                missing_blobs += 1
            else:
                # ``evolutionary.cosine`` takes a Sequence[float]; the blob store
                # hands back a numpy array, whose truthiness it cannot test.
                embeddings[chunk.content_hash] = [float(value) for value in vector]
        if members:
            by_version[version_id] = members

    if not by_version:
        return {}, notes
    if missing_blobs:
        notes.append(
            f"{missing_blobs} chunk(s) had no cached embedding; those cannot be "
            "compared and stay in single-member families."
        )

    families = build_families(
        by_version,
        embeddings or None,
        settings,
        version_order=version_ids,
        total_versions=len(version_ids),
        with_diffs=True,
    )
    by_chunk: dict[str, SnippetFamily] = {}
    for family in families:
        for member in family.members:
            by_chunk[member.chunk_id] = family
    return by_chunk, notes


def _bar(fraction: float, colour: str) -> str:
    """One inline progress bar as HTML.

    Streamlit's own ``st.progress`` is a block element with fixed styling and no
    colour control; three of them per card, ten cards deep, is a wall. This is a
    span with a width, which is all the widget was ever going to be.
    """
    width = max(0.0, min(1.0, fraction)) * 100.0
    return (
        '<span style="display:inline-block;width:120px;height:9px;border-radius:5px;'
        'background:rgba(128,128,128,0.22);vertical-align:middle;overflow:hidden;">'
        f'<span style="display:block;width:{width:.1f}%;height:100%;background:{colour};">'
        "</span></span>"
    )


def _pill(text: str, colour: str) -> str:
    """A coloured badge. Used for the query type and the pass indicator."""
    return (
        f'<span style="background:{colour};color:#fff;padding:2px 9px;border-radius:11px;'
        f'font-size:0.78rem;font-weight:600;white-space:nowrap;">{text}</span>'
    )


def signal_breakdown_html(result: RetrievalResult, plan: QueryPlan, rrf_k: int) -> str | None:
    """The per-signal panel: rank, weighted term, bar, and the dominant marker.

    This is the one piece of the UI US-11 is actually about, so it shows the
    arithmetic rather than a verdict: a jury member can read a rank off the row,
    read the weight off the plan in the sidebar, and check the term by hand.
    """
    terms = rrf_terms(result.signals, plan.strategy_weights, rrf_k)
    if not terms:
        return None
    top_term = max(terms.values())
    winner = dominant_signal(terms)

    rows: list[str] = []
    for signal in SIGNAL_ORDER:
        rank = result.signals.get(signal)
        weight = plan.strategy_weights.get(signal, 0.0)
        term = terms.get(signal)
        colour = SIGNAL_COLOUR[signal]
        name = signal.value.upper()
        if rank is None:
            rows.append(
                f'<tr><td style="padding:2px 10px 2px 0;color:{colour};font-weight:600;">'
                f"{name}</td>"
                '<td style="padding:2px 10px 2px 0;opacity:0.5;">not returned</td>'
                f'<td style="padding:2px 10px 2px 0;">{_bar(0.0, colour)}</td>'
                '<td style="padding:2px 0;opacity:0.5;">&mdash;</td></tr>'
            )
            continue
        share = (term or 0.0) / top_term if top_term > 0 else 0.0
        mark = " &#9733;" if signal is winner else ""
        rows.append(
            f'<tr><td style="padding:2px 10px 2px 0;color:{colour};font-weight:600;">'
            f"{name}{mark}</td>"
            f'<td style="padding:2px 10px 2px 0;">rank {rank}'
            f'<span style="opacity:0.55;"> &times; w={weight:.2f}</span></td>'
            f'<td style="padding:2px 10px 2px 0;">{_bar(share, colour)}</td>'
            f'<td style="padding:2px 0;font-variant-numeric:tabular-nums;">'
            f"{term:.5f}</td></tr>"
        )
    return f'<table style="border-collapse:collapse;font-size:0.86rem;">{"".join(rows)}</table>'


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_sidebar(st: Any, settings: Settings) -> dict[str, Any]:
    """Draw the controls and return the resolved query options."""
    st.sidebar.markdown("### Axiom")
    st.sidebar.caption("Agentic code retrieval &mdash; dense + sparse + structural")

    default_mode = os.environ.get("AXIOM_UI_BACKEND", BACKEND_AUTO).lower()
    modes = [BACKEND_AUTO, BACKEND_HTTP, BACKEND_LOCAL]
    mode = st.sidebar.selectbox(
        "Retrieval path",
        modes,
        index=modes.index(default_mode) if default_mode in modes else 0,
        help="auto prefers a running `axiom serve`, and falls back to this process.",
    )
    base_url = os.environ.get("AXIOM_API_BASE_URL") or DEFAULT_API_BASE_URL
    backend, choice = choose_backend(mode, base_url, settings)

    st.sidebar.markdown(f"**Backend:** {choice.label}")
    for note in choice.notes:
        st.sidebar.info(note)

    version_ids: list[str] = []
    active_version: str | None = None
    index_error: str | None = None
    try:
        listing = backend.versions()
        version_ids = [entry.version_id for entry in listing.versions]
        active_version = listing.active_version
    except ApiError as exc:
        index_error = exc.detail
    except Exception as exc:  # pragma: no cover - defensive
        index_error = f"{type(exc).__name__}: {exc}"

    all_versions = st.sidebar.checkbox(
        "Search every version",
        value=False,
        help="Collapses near-duplicate snippets into families (FR-21).",
        disabled=len(version_ids) < 2,
    )
    options = version_ids or ([active_version] if active_version else [])
    selected_version: str | None = None
    if options and not all_versions:
        labels = [f"{vid}{' (active)' if vid == active_version else ''}" for vid in options]
        index = options.index(active_version) if active_version in options else 0
        selected_version = options[labels.index(st.sidebar.selectbox("Version", labels, index))]

    top_k = st.sidebar.slider("Results", min_value=1, max_value=25, value=settings.top_k_default)
    types = ["auto", *[member.value for member in QueryType]]
    forced = st.sidebar.selectbox("Query type", types, index=0)
    show_families = st.sidebar.checkbox(
        "Load snippet families",
        value=False,
        help="Reads the local index to group each result across versions.",
        disabled=len(version_ids) < 2,
    )

    with st.sidebar.expander("Configuration"):
        st.write(
            {
                "profile": settings.profile,
                "rrf_k": settings.rrf_k,
                "agent_max_passes": settings.agent_max_passes,
                "agent_enabled": settings.agent_enabled,
                "reranker_enabled": settings.reranker_enabled,
                "llm_enabled": settings.llm_enabled,
                "index_root": str(settings.index_root),
            }
        )

    return {
        "backend": backend,
        "choice": choice,
        "index_error": index_error,
        "version": selected_version,
        "all_versions": all_versions,
        "top_k": top_k,
        "query_type": None if forced == "auto" else QueryType(forced),
        "show_families": show_families,
        "version_ids": version_ids,
    }


def _render_header(st: Any, response: QueryResponse, rrf_k: int) -> None:
    """Query-type badge, agent-pass indicator, timing, and the score label."""
    plan = response.query_plan
    colour = QUERY_TYPE_COLOUR.get(plan.query_type, "#666")
    passes = _pill(
        f"{response.passes_used} agent pass{'es' if response.passes_used != 1 else ''}",
        "#555",
    )
    st.markdown(
        f"{_pill(plan.query_type.value.upper(), colour)} &nbsp; {passes} &nbsp; "
        f'<span style="opacity:0.7;">stopped: {response.stop_reason or "n/a"} &middot; '
        f"{response.elapsed_ms:.0f} ms &middot; "
        f"{len(response.results)} result(s)</span>",
        unsafe_allow_html=True,
    )

    weights = " &nbsp; ".join(
        f'<span style="color:{SIGNAL_COLOUR[signal]};font-weight:600;">{signal.value}</span> '
        f"{plan.strategy_weights.get(signal, 0.0):.2f}"
        for signal in SIGNAL_ORDER
    )
    st.markdown(
        f'<span style="font-size:0.86rem;opacity:0.85;">Fusion weights &nbsp; {weights} '
        f"&nbsp;&middot;&nbsp; RRF k={rrf_k} &nbsp;&middot;&nbsp; "
        f"score field: <code>{response.score_field}</code></span>",
        unsafe_allow_html=True,
    )
    if response.score_field == "rrf_score":
        st.caption(
            "The cross-encoder did not run, so `score` is the raw fused RRF value "
            "(bounded by sum(w)/(k+1)), not a calibrated relevance probability."
        )

    if plan.extracted_identifiers or plan.expansion_terms or plan.sub_queries:
        with st.expander("Query plan"):
            if plan.sub_queries:
                st.markdown("**Sub-queries**")
                for sub in plan.sub_queries:
                    st.markdown(f"- {sub}")
            if plan.extracted_identifiers:
                st.markdown(
                    "**Identifiers** &nbsp; "
                    + ", ".join(f"`{name}`" for name in plan.extracted_identifiers)
                )
            if plan.expansion_terms:
                st.markdown(
                    "**Expansion terms** &nbsp; "
                    + ", ".join(f"`{term}`" for term in plan.expansion_terms)
                )


def _render_warnings(st: Any, response: QueryResponse) -> None:
    """Warnings and degradations -- the "what fell back" panel NFR-10 asks for."""
    if not response.warnings and not response.degradations:
        return
    label = f"Degradations and warnings ({len(response.warnings) + len(response.degradations)})"
    with st.expander(label):
        for warning in response.warnings:
            st.warning(f"**{warning.code}** &mdash; {warning.detail}")
        for note in response.degradations:
            st.markdown(f"- {note}")


def _render_timings(st: Any, response: QueryResponse) -> None:
    """Per-stage wall clock. NFR-10's auditability, on screen rather than asserted."""
    if not response.timings:
        return
    with st.expander(f"Stage timings ({response.elapsed_ms:.0f} ms total)"):
        rows = sorted(response.timings.items(), key=lambda item: -item[1])
        longest = max((value for _, value in rows), default=1.0) or 1.0
        lines = ["| stage | ms | |", "|---|---:|---|"]
        for stage, value in rows:
            filled = round(value / longest * 20)
            lines.append(f"| `{stage}` | {value:.1f} | {'&#9608;' * max(filled, 1)} |")
        st.markdown("\n".join(lines), unsafe_allow_html=True)


def _render_family(st: Any, family: SnippetFamily) -> None:
    """The expandable snippet-family view (FR-21, FR-25)."""
    label = (
        f"Snippet family &mdash; {len(family.members)} version(s), stability {family.stability:.2f}"
    )
    with st.expander(label):
        st.markdown(
            "Versions (newest first): " + ", ".join(f"`{version}`" for version in family.versions)
        )
        if not family.is_multi_version:
            st.caption(
                "Single-version family: the stability bonus is exactly neutral here. "
                "A brand-new snippet is novel, not unstable."
            )
        for index, diff in enumerate(family.diffs):
            newer = family.versions[index]
            older = family.versions[index + 1]
            st.markdown(f"**{older} &rarr; {newer}**")
            st.code(diff, language="diff")


def _render_result(
    st: Any,
    position: int,
    result: RetrievalResult,
    response: QueryResponse,
    rrf_k: int,
    movement: dict[str, int],
    family: SnippetFamily | None,
) -> None:
    """One result card: header, score, reasoning, signal breakdown, snippet."""
    chunk = result.chunk
    symbol = chunk.metadata.symbol or chunk.metadata.kind.value
    with st.container(border=True):
        left, right = st.columns([5, 2])
        with left:
            st.markdown(
                f"**{position}. `{chunk.location.as_ref()}`** &nbsp; "
                f'<span style="opacity:0.75;">{symbol} &middot; '
                f"{chunk.metadata.kind.value}</span>",
                unsafe_allow_html=True,
            )
        with right:
            delta = rerank_delta(result, response.query_plan, rrf_k)
            moved = movement.get(chunk.chunk_id, 0)
            caption = f"{response.score_field} &middot; {len(result.signals)}/3 signals"
            if delta is not None and response.score_field != "rrf_score":
                arrow = "&#9650;" if moved > 0 else ("&#9660;" if moved < 0 else "&middot;")
                caption = f"rerank {delta:+.4f} &nbsp; {arrow} {abs(moved) or ''}".strip()
            st.markdown(
                f'<div style="text-align:right;font-size:1.25rem;font-weight:700;'
                f'font-variant-numeric:tabular-nums;">{result.score:.4f}</div>'
                f'<div style="text-align:right;font-size:0.75rem;opacity:0.7;">{caption}</div>',
                unsafe_allow_html=True,
            )

        st.markdown(f"_{result.match_reason}_")

        breakdown = signal_breakdown_html(result, response.query_plan, rrf_k)
        if breakdown is None:
            st.caption("Per-signal breakdown unavailable: the plan carries no matching weights.")
        else:
            st.markdown(breakdown, unsafe_allow_html=True)

        if result.optimization_hint:
            st.info(result.optimization_hint)

        st.code(chunk.text, language=language_for(chunk.location.file_path), line_numbers=False)

        meta: list[str] = [f"`{chunk.chunk_id[:12]}…`", f"version `{chunk.metadata.version_id}`"]
        if chunk.metadata.is_exported:
            meta.append("exported")
        if chunk.metadata.calls:
            meta.append("calls " + ", ".join(f"`{call}`" for call in chunk.metadata.calls[:6]))
        st.caption(" &middot; ".join(meta))

        if family is not None:
            _render_family(st, family)


def _render_legend(st: Any) -> None:
    """One line explaining what the three bars mean. Shown once, above the results."""
    parts = [
        f'<span style="color:{SIGNAL_COLOUR[signal]};font-weight:600;">'
        f"{signal.value}</span> {SIGNAL_GLOSS[signal]}"
        for signal in SIGNAL_ORDER
    ]
    st.markdown(
        '<span style="font-size:0.82rem;opacity:0.8;">Each card shows the rank every '
        "signal gave the chunk and its weighted RRF term "
        "<code>w / (k + rank)</code>; &#9733; marks the dominant signal. &nbsp; "
        + " &nbsp;&middot;&nbsp; ".join(parts)
        + "</span>",
        unsafe_allow_html=True,
    )


def main() -> None:
    """Render the app. The Streamlit entry point.

    Every Streamlit import lives here so the module stays importable on a bare
    install -- ``import axiom.ui.streamlit_app`` must not require Streamlit, both
    because NFR-07 says so and because the helpers above are unit-tested without
    it.
    """
    import streamlit as st

    st.set_page_config(page_title="Axiom", page_icon=":mag:", layout="wide")

    settings = get_settings()
    options = _render_sidebar(st, settings)
    backend = options["backend"]

    st.title("Axiom &mdash; agentic code retrieval")

    with st.form("query"):
        text = st.text_input(
            "Ask about the code",
            placeholder="Which files call preprocessInput before resolveTool?",
        )
        submitted = st.form_submit_button("Search", type="primary")

    if options["index_error"]:
        st.warning(options["index_error"])
        st.markdown(NO_INDEX_HINT)
        return

    if submitted:
        if not text.strip():
            st.error("Enter a query with at least one non-whitespace character.")
        else:
            request = QueryRequest(
                query=text,
                top_k=options["top_k"],
                version=options["version"],
                all_versions=options["all_versions"],
                query_type=options["query_type"],
            )
            try:
                with st.spinner("Retrieving…"):
                    st.session_state["response"] = backend.query(request)
                    st.session_state["families"] = {}
                    st.session_state["family_notes"] = []
            except ApiError as exc:
                st.session_state.pop("response", None)
                st.error(f"**{exc.code.value}** &mdash; {exc.detail}")
                if exc.code is ErrorCode.INDEX_UNAVAILABLE:
                    st.markdown(NO_INDEX_HINT)
            except Exception as exc:  # pragma: no cover - last-resort guard
                st.session_state.pop("response", None)
                _LOG.error("ui query failed: %s", exc, exc_info=True)
                st.error(f"Query failed: {type(exc).__name__}: {exc}")

    response: QueryResponse | None = st.session_state.get("response")
    if response is None:
        st.info(
            'Three archetypes to try: a semantic question ("how is user input '
            'normalised?"), a structural one ("which files call preprocessInput '
            'before resolveTool?"), and a usage one ("where is resolveTool used?").'
        )
        return

    rrf_k = settings.rrf_k
    _render_header(st, response, rrf_k)
    _render_warnings(st, response)
    _render_timings(st, response)

    if not response.results:
        st.warning("No results. Try a broader query, or check the degradation panel above.")
        return

    families: dict[str, SnippetFamily] = st.session_state.get("families", {})
    if options["show_families"] and not families:
        version_ids = options["version_ids"] or response.version_ids
        try:
            families, notes = snippet_families(response.results, settings, version_ids)
        except Exception as exc:  # pragma: no cover - defensive
            _LOG.error("family grouping failed: %s", exc, exc_info=True)
            families, notes = {}, [f"family grouping unavailable: {type(exc).__name__}"]
        st.session_state["families"] = families
        st.session_state["family_notes"] = notes
    for note in st.session_state.get("family_notes", []):
        st.caption(note)

    _render_legend(st)
    movement = rank_movement(response, rrf_k)
    for position, result in enumerate(response.results, start=1):
        _render_result(
            st,
            position,
            result,
            response,
            rrf_k,
            movement,
            families.get(result.chunk.chunk_id) if options["show_families"] else None,
        )


__all__ = [
    "BACKEND_AUTO",
    "BACKEND_HTTP",
    "BACKEND_LOCAL",
    "BackendChoice",
    "HttpBackend",
    "LocalBackend",
    "api_is_up",
    "choose_backend",
    "dominant_signal",
    "language_for",
    "main",
    "rank_movement",
    "reconstructed_rrf",
    "rerank_delta",
    "rrf_terms",
    "signal_breakdown_html",
    "snippet_families",
]


if __name__ == "__main__":
    # Streamlit execs this script with ``__name__ == "__main__"``, so this is the
    # real entry point, not a convenience for direct invocation.
    main()
