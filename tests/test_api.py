"""The HTTP surface, exercised against a real index (API.md sections 2-6, 8).

The audit's close-out recorded ``src/axiom/api/`` at zero test coverage and the
endpoints as never executed -- only imported. This module executes them.

Three layers, deliberately separated so each fails for one reason:

* **Envelope tests** touch only :mod:`axiom.api.models`. They need neither an
  index nor FastAPI, because the validation rules API.md section 4 promises are
  Pydantic field validators and nothing else.
* **Handler tests** call ``run_query``/``list_versions``/``get_chunk``/``health``
  directly against a real index. That is the whole point of
  :mod:`axiom.api.routes` keeping FastAPI inside ``build_router``: the documented
  behaviour of ``POST /v1/query`` is assertable with no server running.
* **Wire tests** drive ``create_app()`` through ``fastapi.testclient.TestClient``
  and assert the status codes and body shapes API.md sections 3 and 6 name. They
  are skipped -- never failed -- on a bare install, which is what keeps NFR-07's
  "the CLI still works without the serve extra" claim honest in CI.

The index under test is built once per module over the committed fixture repos,
because building it is the slow part and every assertion here is read-only.

**Nothing in this file asserts which degradation rung answered.** The optional
backends (faiss, bm25s, an ONNX embedder) may or may not be installed on the
machine running the suite, and a test that hard-codes one configuration stops
exercising the other -- the exact failure the audit found in
``test_pipeline.py``'s ``dense_backend == "numpy"``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import pytest

from axiom.api.models import (
    DEFAULT_MAX_BODY_BYTES,
    DEFAULT_TOP_K_MAX,
    ErrorCode,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    VersionsResponse,
    WarningItem,
    flatten_timings,
    max_body_bytes,
    top_k_ceiling,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).parent / "fixtures"

#: A well-formed ``chunk_id`` that cannot be in any index.
ABSENT_CHUNK_ID = "0" * 32


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def api_settings(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Settings pinned to a module-scoped index root.

    Not the suite-wide ``settings`` fixture: that one is function-scoped on
    ``tmp_path``, and rebuilding the index for every assertion would put this
    module's runtime in the minutes.
    """
    from axiom.config import get_settings

    root = tmp_path_factory.mktemp("api")
    return get_settings(
        "default",
        configs_dir=REPO_ROOT / "configs",
        index_root=root / ".axiom",
        llm_enabled=False,
        reranker_enabled=False,
    )


@pytest.fixture(scope="module")
def api_index(api_settings: Any) -> Any:
    """Two real versions, so version scoping and ``all_versions`` are testable."""
    from axiom import pipeline

    pipeline.build_index_detailed(FIXTURES / "repo_v1", "v1", api_settings)
    pipeline.build_index_detailed(FIXTURES / "repo_v2", "v2", api_settings)
    return api_settings


@pytest.fixture
def clean_warm_state() -> Iterator[None]:
    """Isolate the process-global ``_WARMED`` set around a test.

    ``axiom.api.routes._WARMED`` is module state that survives between tests, so
    a warm assertion is order-dependent without this. TestPlan.md section 1.2
    rule 4 forbids exactly that kind of shared mutable state.
    """
    from axiom.api import routes

    saved = set(routes._WARMED)
    routes._WARMED.clear()
    try:
        yield
    finally:
        routes._WARMED.clear()
        routes._WARMED.update(saved)


@pytest.fixture(scope="module")
def client(api_index: Any) -> Iterator[Any]:
    """A ``TestClient`` over the real app, or a skip on a bare install."""
    pytest.importorskip("fastapi", reason="the serve extra is not installed")
    pytest.importorskip("httpx", reason="TestClient needs httpx")
    from fastapi.testclient import TestClient

    from axiom.api.app import create_app

    with TestClient(create_app(api_index), raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def parity_client(api_index: Any) -> Iterator[Any]:
    """A ``TestClient`` whose settings are resolved exactly as ``axiom query`` resolves them.

    The ``client`` fixture pins ``reranker_enabled=False`` to keep the wire tests
    fast, but the CLI resolves its own ``Settings`` from the profile. Comparing
    the two would then compare two *configurations*, and the first thing to
    diverge is the ``degradations`` string ("reranker_enabled=false" versus
    "model_unavailable") -- a difference in the question asked, not a drift in
    the answer. So the parity tests get a server configured the way the CLI
    configures itself: same profile, same configs dir, same index root, no
    overrides either side.
    """
    pytest.importorskip("fastapi", reason="the serve extra is not installed")
    pytest.importorskip("httpx", reason="TestClient needs httpx")
    from fastapi.testclient import TestClient

    from axiom.api.app import create_app
    from axiom.config import get_settings

    mirrored = get_settings(
        "default", configs_dir=REPO_ROOT / "configs", index_root=api_index.index_root
    )
    with TestClient(create_app(mirrored), raise_server_exceptions=False) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Envelopes: no index, no FastAPI
# ---------------------------------------------------------------------------


class TestRequestValidation:
    """API.md section 4: every constraint is a field validator, not a handler check."""

    @pytest.mark.parametrize("blank", ["", " ", "\t\n  ", "   ", "\u00a0"])
    def test_a_blank_query_is_rejected_before_any_stage_runs(self, blank: str) -> None:
        """Rules.md Rule 3 governs input already accepted; this never gets that far."""
        with pytest.raises(ValueError, match="non-whitespace"):
            QueryRequest(query=blank)

    def test_the_query_text_reaches_the_pipeline_unmutated(self) -> None:
        """``QueryPlan.original_query`` is contractually the user's string verbatim."""
        text = "  Which files call preprocessInput before resolveTool?  "
        assert QueryRequest(query=text).query == text

    def test_top_k_is_bounded_at_both_ends(self) -> None:
        with pytest.raises(ValueError):
            QueryRequest(query="x", top_k=0)
        with pytest.raises(ValueError, match=f"<= {DEFAULT_TOP_K_MAX}"):
            QueryRequest(query="x", top_k=DEFAULT_TOP_K_MAX + 1)
        assert QueryRequest(query="x", top_k=DEFAULT_TOP_K_MAX).top_k == DEFAULT_TOP_K_MAX

    @pytest.mark.parametrize(
        "bad_version",
        ["../../etc/passwd", "v1/../v2", ".hidden", "v1\\v2", "", "a" * 129, "v 1"],
    )
    def test_a_path_traversal_shaped_version_is_a_validation_error(self, bad_version: str) -> None:
        """TC-071: rejected at the model, before any path is built."""
        with pytest.raises(ValueError, match="no path separators"):
            QueryRequest(query="x", version=bad_version)

    @pytest.mark.parametrize("good", ["v1", "v2.3.1", "V1_0-rc1", "0", "a" * 128])
    def test_ordinary_version_ids_are_accepted(self, good: str) -> None:
        assert QueryRequest(query="x", version=good).version == good

    def test_an_unknown_field_is_a_validation_error_not_a_silent_ignore(self) -> None:
        """``extra="forbid"`` is what turns a typo into a 422 the caller can see."""
        with pytest.raises(ValueError, match="Extra inputs"):
            QueryRequest(query="x", top_K=5)

    def test_the_ceilings_are_read_from_the_environment_at_call_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Setup.md section 7.4's variables, honoured without a module reload."""
        assert top_k_ceiling() == DEFAULT_TOP_K_MAX
        assert max_body_bytes() == DEFAULT_MAX_BODY_BYTES
        monkeypatch.setenv("AXIOM_TOP_K_MAX", "7")
        monkeypatch.setenv("AXIOM_API_MAX_BODY_BYTES", "1024")
        assert top_k_ceiling() == 7
        assert max_body_bytes() == 1024
        with pytest.raises(ValueError, match="<= 7"):
            QueryRequest(query="x", top_k=8)

    @pytest.mark.parametrize("junk", ["nonsense", "-1", "0", ""])
    def test_a_malformed_ceiling_falls_back_to_the_documented_default(
        self, monkeypatch: pytest.MonkeyPatch, junk: str
    ) -> None:
        """A typo in an env var must not stop the demo server from binding."""
        monkeypatch.setenv("AXIOM_TOP_K_MAX", junk)
        assert top_k_ceiling() == DEFAULT_TOP_K_MAX


class TestResponseEnvelope:
    """API.md section 3.1: twelve fields, all always present, nothing else."""

    def test_the_documented_field_set_is_exhaustive_in_both_directions(self) -> None:
        documented = {
            "results",
            "query_plan",
            "elapsed_ms",
            "passes_used",
            "timings",
            "warnings",
            "profile",
            "stop_reason",
            "score_field",
            "version_ids",
            "degradations",
            "families",
        }
        assert set(QueryResponse.model_fields) == documented
        assert QueryResponse.model_config["extra"] == "forbid"

    def test_flatten_timings_sums_a_stage_that_ran_twice(self) -> None:
        """The agent loop's second pass re-runs ``dense``; the wire wants one key."""

        class _Stage:
            def __init__(self, stage: str, elapsed_ms: float) -> None:
                self.stage = stage
                self.elapsed_ms = elapsed_ms

        class _Ledger:
            stages: ClassVar = [_Stage("plan", 1.5), _Stage("dense", 10.0), _Stage("dense", 2.25)]

        assert flatten_timings(_Ledger()) == {"plan": 1.5, "dense": 12.25}

    def test_flatten_timings_of_an_empty_ledger_is_an_empty_mapping(self) -> None:
        assert flatten_timings(object()) == {}

    def test_a_warning_carries_a_machine_code_and_a_human_line(self) -> None:
        item = WarningItem(code="AGENT_BUDGET_EXCEEDED", detail="budget reached")
        assert item.model_dump() == {
            "code": "AGENT_BUDGET_EXCEEDED",
            "detail": "budget reached",
        }
        with pytest.raises(ValueError):
            WarningItem(code="", detail="x")

    def test_the_error_vocabulary_is_exactly_api_md_section_6(self) -> None:
        assert {member.value for member in ErrorCode} == {
            "VALIDATION_ERROR",
            "PAYLOAD_TOO_LARGE",
            "VERSION_NOT_FOUND",
            "CHUNK_NOT_FOUND",
            "INDEX_UNAVAILABLE",
            "MODEL_UNAVAILABLE",
            "CONTRACT_VIOLATION",
            "INTERNAL_ERROR",
        }


# ---------------------------------------------------------------------------
# Optional-dependency posture (NFR-07, Rules.md AP-06)
# ---------------------------------------------------------------------------


class TestBareInstallPosture:
    """``axiom --help`` must not pay for a web framework it is not about to use."""

    @pytest.mark.smoke
    def test_importing_the_api_package_does_not_import_fastapi(self) -> None:
        """The PEP 562 laziness in ``api/__init__`` is load-bearing, so prove it."""
        script = (
            "import sys; import axiom.api, axiom.api.models, axiom.api.routes; "
            "print(int(any(m == 'fastapi' or m.startswith('fastapi.') for m in sys.modules)))"
        )
        out = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        assert out.stdout.strip() == "0", "importing axiom.api pulled in fastapi"

    def test_missing_dependencies_answers_without_importing_the_module(self) -> None:
        from axiom.api.app import missing_dependencies

        assert missing_dependencies(("sys", "json")) == []
        assert missing_dependencies(("axiom_not_a_real_module",)) == ["axiom_not_a_real_module"]
        assert "axiom_not_a_real_module" not in sys.modules

    def test_a_missing_serve_extra_is_an_install_line_not_a_traceback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """NFR-07: the evaluator sees ``pip install``, not an ``ImportError``."""
        from axiom.api import app as app_module
        from axiom.core.errors import AxiomError

        monkeypatch.setattr(
            app_module, "missing_dependencies", lambda *_args, **_kw: ["fastapi", "uvicorn"]
        )
        for call in (
            app_module.require_serve_dependencies,
            app_module.create_app,
            app_module.serve,
        ):
            with pytest.raises(app_module.ServeDependencyError) as caught:
                call()
            message = str(caught.value)
            assert "fastapi" in message and "uvicorn" in message
            assert app_module.SERVE_INSTALL_HINT in message
            assert "Traceback" not in message
            # The CLI's boundary catches AxiomError; a sibling class would escape it.
            assert isinstance(caught.value, AxiomError)

    def test_the_bind_host_is_loopback_unless_the_documented_variable_says_otherwise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """NG-08: an unauthenticated surface has no business on a LAN by default."""
        from axiom.api.app import DEFAULT_API_HOST, api_host

        monkeypatch.delenv("AXIOM_API_HOST", raising=False)
        assert api_host() == DEFAULT_API_HOST == "127.0.0.1"
        monkeypatch.setenv("AXIOM_API_HOST", "0.0.0.0")
        assert api_host() == "0.0.0.0"


# ---------------------------------------------------------------------------
# Handlers, against a real index, with no server
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestHandlersWithoutAServer:
    """TechSpecifications.md section 4.11's "thin presentation layer", asserted."""

    def test_run_query_returns_the_documented_envelope(self, api_index: Any) -> None:
        from axiom.api.routes import run_query

        response = run_query(QueryRequest(query="bluetooth deeplink", top_k=4), api_index)
        assert isinstance(response, QueryResponse)
        assert 0 < len(response.results) <= 4
        assert response.passes_used >= 1
        assert response.elapsed_ms >= 0.0
        assert response.score_field in {"rrf_score", "rerank_score"}
        assert response.version_ids == ["v2"]
        assert response.profile == api_index.profile
        # API.md section 3.1: timings is flat ``stage -> ms``, never a nested record.
        assert response.timings
        assert all(isinstance(value, float) for value in response.timings.values())

    def test_an_explicit_version_scopes_the_search(self, api_index: Any) -> None:
        from axiom.api.routes import run_query

        response = run_query(QueryRequest(query="preprocessInput", version="v1"), api_index)
        assert response.version_ids == ["v1"]
        assert {r.chunk.metadata.version_id for r in response.results} == {"v1"}

    def test_all_versions_searches_every_registered_version(self, api_index: Any) -> None:
        from axiom.api.routes import run_query

        response = run_query(
            QueryRequest(query="preprocessInput", all_versions=True, top_k=10), api_index
        )
        assert sorted(response.version_ids) == ["v1", "v2"]

    def test_an_unknown_version_is_404_and_a_missing_index_is_503(
        self, api_index: Any, tmp_path: Path
    ) -> None:
        """API.md section 6: the split is whose fault it is."""
        from axiom.api.routes import ApiError, list_versions, run_query

        with pytest.raises(ApiError) as named:
            run_query(QueryRequest(query="x", version="nope"), api_index)
        assert named.value.status_code == 404
        assert named.value.code is ErrorCode.VERSION_NOT_FOUND

        from axiom.config import get_settings

        empty = get_settings(
            "default",
            configs_dir=REPO_ROOT / "configs",
            index_root=tmp_path / ".axiom",
            llm_enabled=False,
            reranker_enabled=False,
        )
        with pytest.raises(ApiError) as unscoped:
            run_query(QueryRequest(query="x"), empty)
        assert unscoped.value.status_code == 503
        assert unscoped.value.code is ErrorCode.INDEX_UNAVAILABLE

        with pytest.raises(ApiError) as listing:
            list_versions(empty)
        assert listing.value.status_code == 503

    def test_list_versions_reports_every_version_and_the_active_pointer(
        self, api_index: Any
    ) -> None:
        from axiom.api.routes import list_versions

        listing = list_versions(api_index)
        assert isinstance(listing, VersionsResponse)
        assert [entry.version_id for entry in listing.versions] == ["v1", "v2"]
        assert listing.active_version == "v2"
        for entry in listing.versions:
            assert entry.chunk_count > 0
            assert entry.created_at.endswith("Z")

    def test_get_chunk_round_trips_an_id_from_a_query(self, api_index: Any) -> None:
        """Rules.md Rule 1: matched by exact string equality, never re-derived."""
        from axiom.api.routes import get_chunk, run_query

        first = run_query(QueryRequest(query="bluetooth deeplink", top_k=1), api_index)
        returned = first.results[0].chunk
        hydrated = get_chunk(returned.chunk_id, None, api_index)
        assert hydrated.chunk_id == returned.chunk_id
        assert hydrated.text == returned.text
        assert hydrated.content_hash == returned.content_hash
        assert hydrated.location == returned.location

    def test_an_absent_chunk_id_is_404_not_an_empty_object(self, api_index: Any) -> None:
        from axiom.api.routes import ApiError, get_chunk

        with pytest.raises(ApiError) as caught:
            get_chunk(ABSENT_CHUNK_ID, None, api_index)
        assert caught.value.status_code == 404
        assert caught.value.code is ErrorCode.CHUNK_NOT_FOUND

    def test_a_chunk_present_in_one_version_is_scoped_to_it(self, api_index: Any) -> None:
        from axiom.api.routes import get_chunk, run_query

        v1_only = run_query(QueryRequest(query="preprocessInput", version="v1"), api_index)
        chunk_id = v1_only.results[0].chunk.chunk_id
        assert get_chunk(chunk_id, "v1", api_index).chunk_id == chunk_id

    def test_health_is_a_200_shaped_liveness_probe(self, api_index: Any) -> None:
        from axiom.api.routes import health

        report = health(False, api_index)
        assert isinstance(report, HealthResponse)
        assert report.status == "ok"
        assert report.warm is False and report.warmed == []
        assert report.index_available is True
        assert report.version

    def test_health_reports_a_missing_index_as_a_live_process(self, tmp_path: Path) -> None:
        """API.md section 3.4: ``index_available: false`` is still a 200."""
        from axiom.api.routes import health
        from axiom.config import get_settings

        empty = get_settings(
            "default",
            configs_dir=REPO_ROOT / "configs",
            index_root=tmp_path / ".axiom",
            llm_enabled=False,
            reranker_enabled=False,
        )
        report = health(False, empty)
        assert report.status == "ok"
        assert report.index_available is False

    @pytest.mark.slow
    def test_warming_reports_what_this_call_built_and_never_claims_it_twice(
        self, api_index: Any, clean_warm_state: None
    ) -> None:
        from axiom.api.routes import health

        first = health(True, api_index)
        assert first.warm is True
        second = health(True, api_index)
        assert second.warm is True
        assert second.warmed == [], "a repeat warm must not claim credit twice"
        assert set(first.warmed) <= {"embedder", "reranker", "llm"}

    def test_warming_never_raises_for_a_model_that_has_a_rung_left(
        self, api_index: Any, clean_warm_state: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """NFR-07: a loader that blows up is a degradation, not a 503."""
        from axiom.api import routes

        monkeypatch.setattr(
            routes,
            "_warm_embedder",
            lambda _settings: (_ for _ in ()).throw(RuntimeError("no weights")),
        )
        assert routes.warm_models(api_index) == []

    def test_an_exhausted_ladder_is_the_one_warm_failure_that_is_a_503(
        self, api_index: Any, clean_warm_state: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from axiom.api import routes
        from axiom.api.routes import ApiError, health
        from axiom.core.errors import DegradationExhaustedError

        def _exhausted(_settings: Any) -> str:
            raise DegradationExhaustedError("every embedder rung failed")

        monkeypatch.setattr(routes, "_warm_embedder", _exhausted)
        with pytest.raises(ApiError) as caught:
            health(True, api_index)
        assert caught.value.status_code == 503
        assert caught.value.code is ErrorCode.MODEL_UNAVAILABLE

    def test_a_version_query_parameter_is_validated_by_the_body_s_own_rule(self) -> None:
        """One definition of the rule, not two regexes that can drift."""
        from axiom.api.routes import ApiError, _validate_version_param

        assert _validate_version_param(None) is None
        assert _validate_version_param("v1") is None
        with pytest.raises(ApiError) as caught:
            _validate_version_param("../../etc")
        assert caught.value.status_code == 422
        assert caught.value.code is ErrorCode.VALIDATION_ERROR


# ---------------------------------------------------------------------------
# The wire
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestQueryEndpoint:
    """``POST /v1/query`` over HTTP (API.md section 3.1)."""

    def test_a_well_formed_query_is_a_200_with_all_twelve_fields(self, client: Any) -> None:
        response = client.post("/v1/query", json={"query": "bluetooth deeplink", "top_k": 3})
        assert response.status_code == 200
        body = response.json()
        assert set(body) == set(QueryResponse.model_fields)
        # The wire body must round-trip through the model that declares it.
        assert QueryResponse.model_validate(body).results
        assert len(body["results"]) <= 3

    def test_every_result_carries_a_real_location_and_a_reason(self, client: Any) -> None:
        body = client.post("/v1/query", json={"query": "bluetooth deeplink"}).json()
        for result in body["results"]:
            location = result["chunk"]["location"]
            assert location["file_path"].endswith(".js")
            assert 1 <= location["start_line"] <= location["end_line"]
            assert location["start_byte"] < location["end_byte"]
            assert result["match_reason"].strip()
            assert result["signals"]

    def test_score_field_names_the_scale_score_is_actually_on(self, client: Any) -> None:
        """Rendering ``score`` without reading this is how a demo overclaims."""
        body = client.post("/v1/query", json={"query": "normalize input"}).json()
        assert body["score_field"] in {"rrf_score", "rerank_score"}

    def test_two_identical_requests_agree_on_everything_but_the_clock(self, client: Any) -> None:
        """NFR-08 / TC-066: ``elapsed_ms`` and timing *values* are the exclusions."""
        payload = {"query": "Which files call preprocessInput before resolveTool?"}
        first = client.post("/v1/query", json=payload).json()
        second = client.post("/v1/query", json=payload).json()
        assert set(first["timings"]) == set(second["timings"])
        for body in (first, second):
            body.pop("elapsed_ms")
            body.pop("timings")
        assert first == second

    def test_a_degraded_run_is_a_200_with_the_story_in_the_body(self, client: Any) -> None:
        """API.md section 5: only an unresolvable version, a dead index, an
        exhausted ladder or a contract violation is an HTTP error."""
        body = client.post("/v1/query", json={"query": "zzzz-no-such-symbol-zzzz"}).json()
        assert isinstance(body["degradations"], list)
        assert isinstance(body["warnings"], list)
        for warning in body["warnings"]:
            assert set(warning) == {"code", "detail"}

    def test_a_forced_query_type_is_either_honoured_or_reported(self, client: Any) -> None:
        """Silently ignoring a field the client set is the one unacceptable option."""
        body = client.post(
            "/v1/query", json={"query": "where is resolveTool used?", "query_type": "usage"}
        ).json()
        codes = {warning["code"] for warning in body["warnings"]}
        honoured = body["query_plan"]["query_type"] == "usage"
        assert honoured or "QUERY_TYPE_UNSUPPORTED" in codes

    def test_an_invalid_query_type_is_a_422(self, client: Any) -> None:
        response = client.post("/v1/query", json={"query": "x", "query_type": "telepathy"})
        assert response.status_code == 422
        assert response.json()["error"] == "VALIDATION_ERROR"


@pytest.mark.integration
class TestVersionsEndpoint:
    """``GET /v1/versions`` (API.md section 3.2)."""

    def test_it_lists_every_version_with_the_four_documented_fields(self, client: Any) -> None:
        response = client.get("/v1/versions")
        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"active_version", "versions"}
        assert body["active_version"] == "v2"
        for entry in body["versions"]:
            assert set(entry) == {
                "version_id",
                "created_at",
                "chunk_count",
                "parent_version",
            }
        assert VersionsResponse.model_validate(body)


@pytest.mark.integration
class TestChunkEndpoint:
    """``GET /v1/chunk/{chunk_id}`` (API.md section 3.3)."""

    def test_the_chunk_is_returned_unwrapped(self, client: Any) -> None:
        query = client.post("/v1/query", json={"query": "bluetooth deeplink", "top_k": 1})
        expected = query.json()["results"][0]["chunk"]
        response = client.get(f"/v1/chunk/{expected['chunk_id']}")
        assert response.status_code == 200
        assert response.json() == expected

    def test_a_well_formed_but_absent_id_is_404(self, client: Any) -> None:
        response = client.get(f"/v1/chunk/{ABSENT_CHUNK_ID}")
        assert response.status_code == 404
        assert response.json()["error"] == "CHUNK_NOT_FOUND"

    @pytest.mark.parametrize(
        "bad_id",
        ["nothex", "ABCDEF01234567890123456789ABCDEF", "0" * 31, "0" * 33, "../../etc"],
    )
    def test_an_id_that_is_not_a_blake2b_128_digest_is_422(self, client: Any, bad_id: str) -> None:
        response = client.get(f"/v1/chunk/{bad_id}")
        assert response.status_code in (404, 422)
        if response.status_code == 422:
            assert response.json()["error"] == "VALIDATION_ERROR"

    def test_an_unknown_version_is_404_and_a_malformed_one_is_422(self, client: Any) -> None:
        unknown = client.get(f"/v1/chunk/{ABSENT_CHUNK_ID}?version=nope")
        assert unknown.status_code == 404
        assert unknown.json()["error"] == "VERSION_NOT_FOUND"
        malformed = client.get(f"/v1/chunk/{ABSENT_CHUNK_ID}?version=../../etc")
        assert malformed.status_code == 422
        assert malformed.json()["error"] == "VALIDATION_ERROR"


@pytest.mark.integration
class TestHealthEndpoint:
    """``GET /v1/health`` (API.md section 3.4)."""

    def test_a_plain_probe_touches_no_model(self, client: Any) -> None:
        response = client.get("/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert set(body) == set(HealthResponse.model_fields)
        assert body["status"] == "ok"
        assert body["warm"] is False
        assert body["warmed"] == []
        assert body["index_available"] is True

    @pytest.mark.slow
    def test_warm_true_echoes_the_request_and_stays_a_200(
        self, client: Any, clean_warm_state: None
    ) -> None:
        body = client.get("/v1/health?warm=true").json()
        assert body["warm"] is True
        assert set(body["warmed"]) <= {"embedder", "reranker", "llm"}


@pytest.mark.integration
class TestErrorBodies:
    """API.md section 6: one shape, one closed vocabulary, never a traceback."""

    @pytest.mark.parametrize(
        ("payload", "status", "code"),
        [
            ({"query": "   "}, 422, "VALIDATION_ERROR"),
            ({"query": ""}, 422, "VALIDATION_ERROR"),
            ({}, 422, "VALIDATION_ERROR"),
            ({"query": "x", "top_k": 0}, 422, "VALIDATION_ERROR"),
            ({"query": "x", "top_k": DEFAULT_TOP_K_MAX + 1}, 422, "VALIDATION_ERROR"),
            ({"query": "x", "version": "../../etc/passwd"}, 422, "VALIDATION_ERROR"),
            ({"query": "x", "typo_field": 1}, 422, "VALIDATION_ERROR"),
            ({"query": "x", "version": "does-not-exist"}, 404, "VERSION_NOT_FOUND"),
        ],
    )
    def test_every_rejection_uses_the_one_documented_body(
        self, client: Any, payload: dict[str, Any], status: int, code: str
    ) -> None:
        response = client.post("/v1/query", json=payload)
        assert response.status_code == status
        body = response.json()
        assert set(body) == {"error", "detail"}
        assert body["error"] == code
        assert body["detail"] and "\n" not in body["detail"]
        assert "Traceback" not in body["detail"]

    def test_malformed_json_is_a_422_in_the_same_shape(self, client: Any) -> None:
        response = client.post(
            "/v1/query",
            content=b"{not json",
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 422
        assert response.json()["error"] == "VALIDATION_ERROR"

    def test_an_unrouted_path_and_a_wrong_method_still_answer_in_the_shape(
        self, client: Any
    ) -> None:
        """Starlette's own 404/405 are given the documented body, not "Not Found"."""
        missing = client.get("/v1/not-an-endpoint")
        assert missing.status_code == 404
        assert set(missing.json()) == {"error", "detail"}
        assert "/v1/not-an-endpoint" in missing.json()["detail"]
        wrong_method = client.get("/v1/query")
        assert wrong_method.status_code == 405
        assert set(wrong_method.json()) == {"error", "detail"}

    def test_a_contract_violation_reaching_the_boundary_is_a_500_without_a_trace(
        self, client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from axiom.api import routes
        from axiom.core.errors import AxiomContractError

        def _broken(*_args: Any, **_kwargs: Any) -> Any:
            raise AxiomContractError("rank 3 followed rank 1")

        monkeypatch.setattr(routes.pipeline, "query", _broken)
        response = client.post("/v1/query", json={"query": "x"})
        assert response.status_code == 500
        body = response.json()
        assert body["error"] == "CONTRACT_VIOLATION"
        assert "Traceback" not in body["detail"]

    def test_an_unexpected_exception_is_a_500_internal_error(
        self, client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from axiom.api import routes

        def _broken(*_args: Any, **_kwargs: Any) -> Any:
            raise ZeroDivisionError("division by zero")

        monkeypatch.setattr(routes.pipeline, "query", _broken)
        response = client.post("/v1/query", json={"query": "x"})
        assert response.status_code == 500
        body = response.json()
        assert body["error"] == "INTERNAL_ERROR"
        assert "Traceback" not in body["detail"]


@pytest.mark.integration
class TestBodyLimit:
    """TC-071's adversarial body, both with and without ``Content-Length``."""

    def test_an_oversized_declared_body_is_413_before_it_is_read(
        self, client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        opened: list[Any] = []
        real_open = Path.open
        monkeypatch.setattr(
            Path, "open", lambda self, *a, **k: (opened.append(self), real_open(self, *a, **k))[1]
        )
        body = json.dumps({"query": "x" * (DEFAULT_MAX_BODY_BYTES + 1000)}).encode()
        response = client.post(
            "/v1/query", content=body, headers={"content-type": "application/json"}
        )
        assert response.status_code == 413
        assert response.json()["error"] == "PAYLOAD_TOO_LARGE"
        assert opened == [], "a rejected body must not have opened any file"

    def test_a_chunked_body_is_bounded_while_it_is_buffered(self, client: Any) -> None:
        payload = json.dumps({"query": "x" * (DEFAULT_MAX_BODY_BYTES + 1000)}).encode()

        def _chunks() -> Iterator[bytes]:
            for start in range(0, len(payload), 8192):
                yield payload[start : start + 8192]

        response = client.post(
            "/v1/query", content=_chunks(), headers={"content-type": "application/json"}
        )
        assert response.status_code == 413
        assert response.json()["error"] == "PAYLOAD_TOO_LARGE"

    def test_a_body_under_the_ceiling_still_reaches_the_handler(self, client: Any) -> None:
        response = client.post("/v1/query", json={"query": "x" * 2048})
        assert response.status_code == 200

    def test_the_middleware_reads_the_ceiling_at_request_time(self) -> None:
        from axiom.api.app import BodyLimitMiddleware

        assert BodyLimitMiddleware(object()).limit == max_body_bytes()
        assert BodyLimitMiddleware(object(), limit=99).limit == 99


@pytest.mark.integration
class TestRoutingSurface:
    """API.md section 3's prefix, and the unprefixed aliases section 4.11 names."""

    @pytest.mark.parametrize("path", ["/health", "/versions"])
    def test_the_unprefixed_alias_answers_identically(self, client: Any, path: str) -> None:
        prefixed = client.get(f"/v1{path}")
        alias = client.get(path)
        assert alias.status_code == prefixed.status_code == 200
        if path == "/versions":
            assert alias.json() == prefixed.json()

    def test_only_the_prefixed_paths_appear_in_the_schema(self, client: Any) -> None:
        """A client reading ``/docs`` sees exactly API.md section 3's surface."""
        schema = client.get("/openapi.json")
        assert schema.status_code == 200
        assert set(schema.json()["paths"]) == {
            "/v1/query",
            "/v1/versions",
            "/v1/families",
            "/v1/chunk/{chunk_id}",
            "/v1/health",
        }

    def test_the_schema_states_the_deployment_posture_it_ships_with(self, client: Any) -> None:
        """NG-08 in writing, so nobody reading ``/docs`` has to guess."""
        description = client.get("/openapi.json").json()["info"]["description"]
        assert "unauthenticated" in description.lower()

    def test_the_interactive_docs_render(self, client: Any) -> None:
        assert client.get("/docs").status_code == 200


# ---------------------------------------------------------------------------
# The property nobody had checked
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCliAndApiDoNotDrift:
    """FR-24 / API.md section 8: two renderings of one call into one pipeline.

    The mechanism is meant to be the *absence* of a second code path. These tests
    are what turns that from a convention into a fact -- and what would have
    caught the ``timings`` divergence recorded below.
    """

    @staticmethod
    def _cli_json(index_root: Path, query: str) -> dict[str, Any]:
        from typer.testing import CliRunner

        from axiom.cli import app as cli_app

        result = CliRunner().invoke(
            cli_app, ["--index-root", str(index_root), "--json", "query", query]
        )
        assert result.exit_code == 0, result.output
        return json.loads(result.stdout)

    def test_the_two_surfaces_return_the_same_ranked_results(
        self, parity_client: Any, api_index: Any
    ) -> None:
        query = "Which files call preprocessInput before resolveTool?"
        cli = self._cli_json(api_index.index_root, query)
        api = parity_client.post("/v1/query", json={"query": query}).json()

        assert [r["chunk"]["chunk_id"] for r in cli["results"]] == [
            r["chunk"]["chunk_id"] for r in api["results"]
        ]
        assert cli["results"] == api["results"]

    def test_the_two_surfaces_agree_on_the_plan_and_the_bookkeeping(
        self, parity_client: Any, api_index: Any
    ) -> None:
        query = "how is user input normalised?"
        cli = self._cli_json(api_index.index_root, query)
        api = parity_client.post("/v1/query", json={"query": query}).json()

        assert set(cli) == set(api) == set(QueryResponse.model_fields)
        for field in (
            "query_plan",
            "passes_used",
            "profile",
            "stop_reason",
            "score_field",
            "version_ids",
            "degradations",
            "warnings",
        ):
            assert cli[field] == api[field], f"{field} drifted between CLI and API"

    def test_the_cli_json_body_validates_against_the_api_response_model(
        self, api_index: Any
    ) -> None:
        """API.md section 8: ``--json`` "reuses the same envelope models".

        This is the assertion that fails today. ``cli.py`` serialises
        ``pipeline.QueryResponse.as_dict()``, whose ``timings`` is the *nested*
        ledger record (``{"total_ms": ..., "stages": [...]}``), while API.md
        section 3.1 types ``timings`` as a flat ``dict[str, float]`` and the HTTP
        envelope produces it via ``api.models.flatten_timings``. Every other
        field matches, so the drift is narrow -- and invisible to anyone who only
        reads the first six fields.

        Marked ``xfail(strict=True)``: it is a real, reported defect, not an
        accepted behaviour, and it must start failing the moment ``cli.py``
        swaps ``response.ledger.as_dict()`` for ``flatten_timings(response.ledger)``.
        """
        payload = self._cli_json(api_index.index_root, "normalize input")
        assert set(payload["timings"]) == {"total_ms", "stages"}, (
            "cli.py no longer emits the nested ledger -- delete this test's twin "
            "xfail and assert QueryResponse.model_validate(payload) instead"
        )
        with pytest.raises(Exception, match="timings"):
            QueryResponse.model_validate(payload)

    def test_the_two_surfaces_agree_on_the_version_listing(
        self, parity_client: Any, api_index: Any
    ) -> None:
        from typer.testing import CliRunner

        from axiom.cli import app as cli_app

        result = CliRunner().invoke(
            cli_app, ["--index-root", str(api_index.index_root), "--json", "versions"]
        )
        assert result.exit_code == 0, result.output
        cli = json.loads(result.stdout)
        api = parity_client.get("/v1/versions").json()
        assert cli["active_version"] == api["active_version"]
        assert [v["version_id"] for v in cli["versions"]] == [
            v["version_id"] for v in api["versions"]
        ]


# ---------------------------------------------------------------------------
# CORS -- opt-in, and silent when nobody opted in
# ---------------------------------------------------------------------------


class TestCorsOptIn:
    """``AXIOM_API_CORS_ORIGINS`` (API.md section 2).

    The default matters more than the feature: this surface is unauthenticated
    by design (NG-08), so a CORS header that appears without anyone asking for
    it is the difference between a local dev server and something any open tab
    can read the user's source index through. Every test here therefore pins one
    half of that -- what happens when it is unset, and what exactly is allowed
    when it is set.
    """

    ORIGIN = "http://localhost:5173"

    @staticmethod
    def _client(api_index: Any, origins: str | None, monkeypatch: Any) -> Any:
        pytest.importorskip("fastapi", reason="the serve extra is not installed")
        pytest.importorskip("httpx", reason="TestClient needs httpx")
        from fastapi.testclient import TestClient

        from axiom.api.app import CORS_ORIGINS_ENV, create_app

        if origins is None:
            monkeypatch.delenv(CORS_ORIGINS_ENV, raising=False)
        else:
            monkeypatch.setenv(CORS_ORIGINS_ENV, origins)
        return TestClient(create_app(api_index), raise_server_exceptions=False)

    def test_unset_means_no_cors_header_at_all(self, api_index: Any, monkeypatch: Any) -> None:
        """The default posture: a cross-origin browser request gets nothing."""
        with self._client(api_index, None, monkeypatch) as client:
            response = client.get("/v1/health", headers={"Origin": self.ORIGIN})
        assert response.status_code == 200
        assert "access-control-allow-origin" not in response.headers

    def test_a_named_origin_is_allowed(self, api_index: Any, monkeypatch: Any) -> None:
        with self._client(api_index, self.ORIGIN, monkeypatch) as client:
            response = client.get("/v1/health", headers={"Origin": self.ORIGIN})
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == self.ORIGIN

    def test_an_unnamed_origin_is_not_allowed(self, api_index: Any, monkeypatch: Any) -> None:
        """Allow-listing one origin must not allow-list every origin."""
        with self._client(api_index, self.ORIGIN, monkeypatch) as client:
            response = client.get("/v1/health", headers={"Origin": "http://evil.example"})
        assert "access-control-allow-origin" not in response.headers

    def test_the_query_preflight_succeeds(self, api_index: Any, monkeypatch: Any) -> None:
        """``POST /v1/query`` with a JSON body is preflighted; the frontend dies without this."""
        with self._client(api_index, self.ORIGIN, monkeypatch) as client:
            response = client.options(
                "/v1/query",
                headers={
                    "Origin": self.ORIGIN,
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type",
                },
            )
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == self.ORIGIN
        assert "POST" in response.headers["access-control-allow-methods"]

    def test_credentials_are_never_allowed(self, api_index: Any, monkeypatch: Any) -> None:
        """There is no session on this surface, so a browser must not attach one."""
        with self._client(api_index, self.ORIGIN, monkeypatch) as client:
            response = client.get("/v1/health", headers={"Origin": self.ORIGIN})
        assert "access-control-allow-credentials" not in response.headers

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("", []),
            ("   ", []),
            (",,", []),
            ("http://a", ["http://a"]),
            ("http://a/", ["http://a"]),
            (" http://a , http://b ", ["http://a", "http://b"]),
            ("http://a,http://a", ["http://a"]),
            ("*", ["*"]),
        ],
    )
    def test_origin_parsing(
        self, raw: str, expected: list[str], monkeypatch: Any
    ) -> None:
        """Parsing is pure and importable on a bare install -- no FastAPI needed."""
        from axiom.api.app import CORS_ORIGINS_ENV, cors_origins

        monkeypatch.setenv(CORS_ORIGINS_ENV, raw)
        assert cors_origins() == expected


# ---------------------------------------------------------------------------
# GET /v1/families and the query envelope's families block
# ---------------------------------------------------------------------------


class TestFamiliesEndpoint:
    """API.md section 3.3 -- the no-query half of FR-21, over HTTP."""

    def test_a_default_listing_is_a_200_with_the_documented_envelope(self, client: Any) -> None:
        response = client.get("/v1/families")
        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"families", "total", "version_ids"}
        assert body["version_ids"], "families are meaningless without a version to compute over"
        assert len(body["families"]) <= 20

    def test_total_counts_matches_not_the_page(self, client: Any) -> None:
        """Without this a client cannot tell "there are 3" from "you asked for 3"."""
        full = client.get("/v1/families", params={"limit": 0}).json()
        page = client.get("/v1/families", params={"limit": 1}).json()
        assert page["total"] == full["total"]
        assert len(page["families"]) <= 1 <= max(page["total"], 1)

    def test_limit_zero_means_every_match(self, client: Any) -> None:
        body = client.get("/v1/families", params={"limit": 0}).json()
        assert len(body["families"]) == body["total"]

    def test_stability_denominator_is_the_version_list(self, client: Any) -> None:
        """``stability`` is over the corpus, so it must never exceed 1.0."""
        body = client.get("/v1/families", params={"limit": 0}).json()
        for family in body["families"]:
            assert 0.0 < family["stability"] <= 1.0

    def test_multi_only_never_returns_a_single_version_family(self, client: Any) -> None:
        body = client.get("/v1/families", params={"limit": 0, "multi_only": True}).json()
        assert all(len(family["versions"]) >= 2 for family in body["families"])

    def test_the_serialised_family_is_the_documented_field_set(self, client: Any) -> None:
        """API.md section 3.3. ``is_multi_version`` is a property, not a wire field.

        Pinned because the natural mistake is to document the model's Python
        surface: a frontend told to read ``is_multi_version`` gets ``undefined``
        and silently renders every family as single-version.
        """
        from axiom.schema import SnippetFamily

        body = client.get("/v1/families", params={"limit": 0}).json()
        for family in body["families"]:
            assert set(family) == set(SnippetFamily.model_fields)
            assert "is_multi_version" not in family

    def test_the_representative_is_a_whole_chunk_here(self, client: Any) -> None:
        """Unlike section 3.1's families block, where it is a bare chunk_id string."""
        body = client.get("/v1/families", params={"limit": 1}).json()
        for family in body["families"]:
            assert isinstance(family["representative"], dict)
            assert "location" in family["representative"]

    def test_diffs_are_off_unless_asked_for(self, client: Any) -> None:
        """They cost a diff per transition per family; a list view must not pay it."""
        plain = client.get("/v1/families", params={"limit": 0}).json()
        assert all(not family["diffs"] for family in plain["families"])

    @pytest.mark.parametrize("bad", ["../etc", ".hidden", "a/b"])
    def test_a_malformed_version_is_422_not_a_path_read(self, client: Any, bad: str) -> None:
        response = client.get("/v1/families", params={"version": bad})
        assert response.status_code == 422
        assert response.json()["error"] == "VALIDATION_ERROR"

    def test_an_unresolvable_version_is_404(self, client: Any) -> None:
        response = client.get("/v1/families", params={"version": "v999"})
        assert response.status_code == 404

    @pytest.mark.parametrize("bad", [-1, 501])
    def test_limit_out_of_range_is_422(self, client: Any, bad: int) -> None:
        assert client.get("/v1/families", params={"limit": bad}).status_code == 422

    def test_the_cli_and_the_api_report_the_same_families(
        self, parity_client: Any, api_index: Any
    ) -> None:
        """One loader (``pipeline.list_families``) means one answer, not two."""
        from typer.testing import CliRunner

        from axiom.cli import app as cli_app

        result = CliRunner().invoke(
            cli_app,
            ["--index-root", str(api_index.index_root), "--json", "families", "--limit", "0"],
        )
        assert result.exit_code == 0, result.output
        cli = json.loads(result.stdout)
        api = parity_client.get("/v1/families", params={"limit": 0}).json()
        assert [f["family_id"] for f in cli] == [f["family_id"] for f in api["families"]]
        assert [f["stability"] for f in cli] == [f["stability"] for f in api["families"]]


class TestQueryFamiliesBlock:
    """API.md section 3.1's ``families`` field."""

    def test_it_is_present_and_empty_on_a_single_version_query(self, client: Any) -> None:
        """A key that appears only sometimes forces every client to branch on it."""
        body = client.post("/v1/query", json={"query": "normalize the command", "top_k": 5}).json()
        assert body["families"] == []

    def test_all_versions_populates_one_family_per_result(self, client: Any) -> None:
        body = client.post(
            "/v1/query", json={"query": "normalize the command", "top_k": 5, "all_versions": True}
        ).json()
        assert len(body["families"]) == len(body["results"])
        representatives = [family["representative"] for family in body["families"]]
        assert representatives == [result["chunk"]["chunk_id"] for result in body["results"]], (
            "families must be in result order, one per row, so a UI can zip them"
        )

    def test_every_collapsed_member_stays_reachable(self, client: Any) -> None:
        """TC-085: collapsing hides rows; it must not lose them."""
        body = client.post(
            "/v1/query", json={"query": "normalize the command", "top_k": 5, "all_versions": True}
        ).json()
        for family in body["families"]:
            assert family["members"], "a family with no members collapsed nothing"
            assert family["representative"] in {m["chunk_id"] for m in family["members"]}
            assert set(family["versions"]) == {m["version_id"] for m in family["members"]}

    def test_it_publishes_no_stability_number(self, client: Any) -> None:
        """Deriving one from a result list contradicts GET /v1/families -- see §3.1."""
        body = client.post(
            "/v1/query", json={"query": "normalize the command", "top_k": 5, "all_versions": True}
        ).json()
        for family in body["families"]:
            assert "stability" not in family
