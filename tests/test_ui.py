"""The Streamlit demo surface (FR-25, TC-072).

The audit recorded ``src/axiom/ui/`` at zero coverage and the app as never
executed -- only imported. Two layers here fix that:

* **Everything above ``main()``** -- the two backends, the RRF reconstruction
  the result cards draw, the launcher -- is ordinary Python with no Streamlit in
  it, by design. Those tests run on a bare install.
* **``main()`` itself** is driven through ``streamlit.testing.v1.AppTest``, which
  executes the script in-process and surfaces any exception the page raised. It
  is the supported headless driver and it is what makes "the UI renders" an
  assertion instead of a claim. Skipped, never failed, when Streamlit is absent.

The one thing these tests deliberately do **not** do is assert on rendered
pixels or on Streamlit's internal element protobufs beyond counting and reading
text. A demo page's job is to not throw and to show the retrieved chunks; a test
that pins its layout would fail on every honest design change.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from axiom.schema import (
    Chunk,
    ChunkKind,
    ChunkLocation,
    ChunkMetadata,
    QueryPlan,
    QueryType,
    RetrievalResult,
    SignalKind,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).parent / "fixtures"
APP_SCRIPT = REPO_ROOT / "src" / "axiom" / "ui" / "streamlit_app.py"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _chunk(text: str = "export function alpha() { return 1; }", path: str = "src/a.js") -> Chunk:
    encoded = text.encode("utf-8")
    return Chunk.create(
        text,
        ChunkLocation(
            file_path=path,
            start_line=1,
            end_line=max(1, text.count("\n") + 1),
            start_byte=0,
            end_byte=len(encoded),
        ),
        ChunkMetadata(
            symbol="alpha",
            kind=ChunkKind.FUNCTION,
            language="javascript",
            version_id="v1",
        ),
    )


def _result(score: float, signals: dict[SignalKind, int], path: str = "src/a.js") -> Any:
    return RetrievalResult(
        chunk=_chunk(path=path),
        score=score,
        match_reason="dense: nearest neighbour",
        signals=signals,
    )


def _plan(weights: dict[SignalKind, float]) -> QueryPlan:
    return QueryPlan(
        original_query="q",
        query_type=QueryType.SEMANTIC,
        strategy_weights=weights,
    )


# ---------------------------------------------------------------------------
# Optional-dependency posture
# ---------------------------------------------------------------------------


class TestBareInstallPosture:
    """NFR-07: ``axiom ui`` without Streamlit prints an install line, not a traceback."""

    @pytest.mark.smoke
    def test_importing_the_app_module_does_not_import_streamlit(self) -> None:
        """Every Streamlit symbol lives inside ``main()`` precisely so this holds."""
        script = (
            "import sys; import axiom.ui, axiom.ui.streamlit_app; "
            "print(int(any(m == 'streamlit' or m.startswith('streamlit.') "
            "for m in sys.modules)))"
        )
        out = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        assert out.stdout.strip() == "0", "importing axiom.ui pulled in streamlit"

    def test_a_missing_streamlit_is_an_install_line(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from axiom import ui
        from axiom.core.errors import AxiomError

        monkeypatch.setattr(ui, "missing_dependencies", lambda: ["streamlit"])
        with pytest.raises(ui.UiDependencyError) as caught:
            ui.require_ui_dependencies()
        assert ui.UI_INSTALL_HINT in str(caught.value)
        assert "Traceback" not in str(caught.value)
        assert isinstance(caught.value, AxiomError)
        with pytest.raises(ui.UiDependencyError):
            ui.launch()

    def test_the_launcher_targets_the_installed_script_in_this_interpreter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The child must stay inside the venv holding the index it is about to read."""
        from axiom import ui

        assert ui.app_path() == APP_SCRIPT
        assert ui.app_path().is_file()

        seen: dict[str, Any] = {}

        def _fake_call(command: list[str], env: dict[str, str]) -> int:
            seen["command"] = command
            seen["env"] = env
            return 0

        monkeypatch.setattr(ui, "require_ui_dependencies", lambda: None)
        monkeypatch.setattr(ui.subprocess, "call", _fake_call)
        assert ui.launch(port=8599, api_base_url_value="http://x:1", headless=True) == 0

        command = seen["command"]
        assert command[0] == sys.executable
        assert command[1:4] == ["-m", "streamlit", "run"]
        assert command[4] == str(ui.app_path())
        assert "--server.port" in command and "8599" in command
        assert command[-2:] == ["--server.headless", "true"]
        assert seen["env"]["AXIOM_API_BASE_URL"] == "http://x:1"
        assert seen["env"]["AXIOM_UI_PORT"] == "8599"

    def test_the_api_base_url_honours_the_documented_variable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from axiom.ui import DEFAULT_API_BASE_URL, api_base_url

        monkeypatch.delenv("AXIOM_API_BASE_URL", raising=False)
        assert api_base_url() == DEFAULT_API_BASE_URL == "http://127.0.0.1:8000"
        monkeypatch.setenv("AXIOM_API_BASE_URL", "http://elsewhere:9000")
        assert api_base_url() == "http://elsewhere:9000"


# ---------------------------------------------------------------------------
# The card arithmetic
# ---------------------------------------------------------------------------


class TestScoreReconstruction:
    """The per-signal breakdown is the demo's explainability claim; check the maths."""

    def test_rrf_terms_are_the_same_arithmetic_fusion_performed(self) -> None:
        from axiom.ui.streamlit_app import rrf_terms

        terms = rrf_terms(
            {SignalKind.DENSE: 1, SignalKind.SPARSE: 4},
            {SignalKind.DENSE: 0.6, SignalKind.SPARSE: 0.4},
            60,
        )
        assert terms[SignalKind.DENSE] == pytest.approx(0.6 / 61)
        assert terms[SignalKind.SPARSE] == pytest.approx(0.4 / 64)

    def test_a_signal_the_plan_gives_no_weight_contributes_nothing(self) -> None:
        """Fusion omitted it; the card must omit it too, not draw a zero bar."""
        from axiom.ui.streamlit_app import rrf_terms

        terms = rrf_terms(
            {SignalKind.DENSE: 1, SignalKind.STRUCTURAL: 2},
            {SignalKind.DENSE: 1.0, SignalKind.STRUCTURAL: 0.0},
            60,
        )
        assert set(terms) == {SignalKind.DENSE}

    def test_an_unreconstructable_score_is_none_not_zero(self) -> None:
        from axiom.ui.streamlit_app import reconstructed_rrf, rerank_delta

        result = _result(0.9, {SignalKind.DENSE: 1})
        empty_plan = _plan({SignalKind.SPARSE: 1.0})
        assert reconstructed_rrf(result, empty_plan, 60) is None
        assert rerank_delta(result, empty_plan, 60) is None

    def test_the_passthrough_rung_shows_exactly_zero_movement(self) -> None:
        from axiom.ui.streamlit_app import rerank_delta

        plan = _plan({SignalKind.DENSE: 1.0})
        rrf = 1.0 / 61
        assert rerank_delta(_result(rrf, {SignalKind.DENSE: 1}), plan, 60) == pytest.approx(0.0)

    def test_dominant_signal_breaks_ties_in_the_documented_order(self) -> None:
        from axiom.ui.streamlit_app import dominant_signal

        tied = {SignalKind.STRUCTURAL: 0.5, SignalKind.SPARSE: 0.5, SignalKind.DENSE: 0.5}
        assert dominant_signal(tied) is SignalKind.DENSE
        assert dominant_signal({SignalKind.SPARSE: 0.9, SignalKind.DENSE: 0.1}) is SignalKind.SPARSE
        assert dominant_signal({}) is None

    def test_rank_movement_is_empty_when_any_result_cannot_be_reconstructed(self) -> None:
        """Half a movement column is worse than none: it reads as "did not move"."""
        from axiom.api.models import QueryResponse
        from axiom.ui.streamlit_app import rank_movement

        plan = _plan({SignalKind.SPARSE: 1.0})
        response = QueryResponse(
            results=[_result(0.9, {SignalKind.DENSE: 1})],
            query_plan=plan,
            elapsed_ms=1.0,
            passes_used=1,
        )
        assert rank_movement(response, 60) == {}

    def test_rank_movement_reports_positions_gained_against_the_first_stage(self) -> None:
        from axiom.api.models import QueryResponse
        from axiom.ui.streamlit_app import rank_movement

        plan = _plan({SignalKind.DENSE: 1.0})
        # ``b`` had the better first-stage rank but is returned second, so the
        # second stage demoted it by one and promoted ``a`` by one.
        first = _result(0.99, {SignalKind.DENSE: 9}, path="src/a.js")
        second = _result(0.50, {SignalKind.DENSE: 1}, path="src/b.js")
        response = QueryResponse(
            results=[first, second], query_plan=plan, elapsed_ms=1.0, passes_used=1
        )
        movement = rank_movement(response, 60)
        assert movement[first.chunk.chunk_id] == 1
        assert movement[second.chunk.chunk_id] == -1

    def test_the_breakdown_html_is_omitted_rather_than_drawn_empty(self) -> None:
        from axiom.ui.streamlit_app import signal_breakdown_html

        no_overlap = _plan({SignalKind.SPARSE: 1.0})
        assert signal_breakdown_html(_result(0.5, {SignalKind.DENSE: 1}), no_overlap, 60) is None
        html = signal_breakdown_html(
            _result(0.5, {SignalKind.DENSE: 1}), _plan({SignalKind.DENSE: 1.0}), 60
        )
        assert html is not None and "dense" in html.lower()

    @pytest.mark.parametrize(
        ("path", "language"),
        [
            ("src/a.js", "javascript"),
            ("src/a.JSX", "javascript"),
            ("src/a.ts", "typescript"),
            ("a.py", "python"),
            ("README.md", "markdown"),
            ("Makefile", "text"),
            ("src/archive.tar.gz", "text"),
        ],
    )
    def test_an_unknown_extension_is_left_unhighlighted(self, path: str, language: str) -> None:
        """Mis-highlighted code reads as broken code on a projector."""
        from axiom.ui.streamlit_app import language_for

        assert language_for(path) == language


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


class TestBackendSelection:
    """``auto`` must prefer a running server and say so when it falls back."""

    def test_local_mode_never_probes_the_network(
        self, settings: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from axiom.ui import streamlit_app as ui_app

        monkeypatch.setattr(
            ui_app,
            "api_is_up",
            lambda *_a, **_kw: pytest.fail("local mode must not probe the API"),
        )
        backend, choice = ui_app.choose_backend("local", "http://127.0.0.1:1", settings)
        assert isinstance(backend, ui_app.LocalBackend)
        assert choice.kind == "local"
        assert choice.notes == []

    def test_http_mode_is_honoured_even_when_nothing_answers(
        self, settings: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pinned choice that silently becomes another choice is not a pin."""
        from axiom.ui import streamlit_app as ui_app

        monkeypatch.setattr(ui_app, "api_is_up", lambda *_a, **_kw: False)
        backend, choice = ui_app.choose_backend("http", "http://127.0.0.1:1", settings)
        assert isinstance(backend, ui_app.HttpBackend)
        assert choice.kind == "http"

    def test_auto_falls_back_to_this_process_and_says_so(
        self, settings: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from axiom.ui import streamlit_app as ui_app

        monkeypatch.setattr(ui_app, "api_is_up", lambda *_a, **_kw: False)
        backend, choice = ui_app.choose_backend("auto", "http://127.0.0.1:1", settings)
        assert isinstance(backend, ui_app.LocalBackend)
        assert choice.notes and "http://127.0.0.1:1" in choice.notes[0]

    def test_auto_prefers_a_live_api(self, settings: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        from axiom.ui import streamlit_app as ui_app

        monkeypatch.setattr(ui_app, "api_is_up", lambda *_a, **_kw: True)
        backend, choice = ui_app.choose_backend("auto", "http://127.0.0.1:8000", settings)
        assert isinstance(backend, ui_app.HttpBackend)
        assert choice.base_url == "http://127.0.0.1:8000"

    def test_the_empty_state_remediation_matches_the_failure_that_produced_it(self) -> None:
        """A refused connection and an unbuilt index share one error code.

        ``HttpBackend`` maps "connection refused" onto ``INDEX_UNAVAILABLE``,
        which is the right status and the wrong advice: printing "run
        ``axiom index``" at someone whose index is fine and whose server is not
        running sends them to rebuild a corpus they already have.
        """
        from axiom.ui.streamlit_app import (
            NO_API_HINT,
            NO_INDEX_HINT,
            BackendChoice,
            _remediation_for,
        )

        http = BackendChoice("http", "HTTP API at http://127.0.0.1:8000")
        local = BackendChoice("local", "in-process pipeline")
        assert _remediation_for(http) == NO_API_HINT
        assert "axiom serve" in NO_API_HINT
        assert _remediation_for(local) == NO_INDEX_HINT
        assert "axiom index" in NO_INDEX_HINT

    def test_a_refused_probe_is_false_and_never_an_exception(self) -> None:
        from axiom.ui.streamlit_app import api_is_up

        # Port 1 is privileged and unbound; the probe must swallow the refusal.
        assert api_is_up("http://127.0.0.1:1", timeout=0.25) is False

    def test_an_unreachable_api_becomes_the_same_error_the_local_path_raises(self) -> None:
        """One exception type, so the rendering code never asks which backend failed."""
        from axiom.api.models import ErrorCode, QueryRequest
        from axiom.api.routes import ApiError
        from axiom.ui.streamlit_app import HttpBackend

        backend = HttpBackend("http://127.0.0.1:1")
        with pytest.raises(ApiError) as caught:
            backend.query(QueryRequest(query="x"))
        assert caught.value.status_code == 503
        assert caught.value.code is ErrorCode.INDEX_UNAVAILABLE

    def test_a_server_error_body_is_decoded_back_into_its_code(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from urllib.error import HTTPError

        from axiom.api.models import ErrorCode, QueryRequest
        from axiom.api.routes import ApiError
        from axiom.ui import streamlit_app as ui_app

        class _Body:
            @staticmethod
            def read() -> bytes:
                return json.dumps(
                    {"error": "VERSION_NOT_FOUND", "detail": "version 'nope' has no manifest"}
                ).encode()

        def _raise(*_args: Any, **_kwargs: Any) -> Any:
            error = HTTPError("http://x/v1/query", 404, "Not Found", {}, None)  # type: ignore[arg-type]
            error.read = _Body.read  # type: ignore[method-assign]
            raise error

        monkeypatch.setattr(ui_app, "urlopen", _raise)
        with pytest.raises(ApiError) as caught:
            ui_app.HttpBackend("http://x").query(QueryRequest(query="x"))
        assert caught.value.status_code == 404
        assert caught.value.code is ErrorCode.VERSION_NOT_FOUND
        assert "nope" in caught.value.detail

    @pytest.mark.integration
    def test_the_local_backend_goes_through_the_api_handlers(
        self, indexed_v1: tuple[Any, Any]
    ) -> None:
        """ "The UI shows something different from the API" must not be possible."""
        from axiom.api.models import QueryRequest
        from axiom.api.routes import run_query
        from axiom.ui.streamlit_app import LocalBackend

        settings, _report = indexed_v1
        request = QueryRequest(query="Where is the Bluetooth-settings deeplink used?", top_k=5)
        through_ui = LocalBackend(settings).query(request)
        through_api = run_query(request, settings)
        assert [r.chunk.chunk_id for r in through_ui.results] == [
            r.chunk.chunk_id for r in through_api.results
        ]
        assert through_ui.score_field == through_api.score_field
        assert through_ui.version_ids == through_api.version_ids

    @pytest.mark.integration
    def test_the_local_backend_lists_the_same_versions_the_api_does(
        self, indexed_v1: tuple[Any, Any]
    ) -> None:
        from axiom.api.routes import list_versions
        from axiom.ui.streamlit_app import LocalBackend

        settings, _report = indexed_v1
        assert LocalBackend(settings).versions() == list_versions(settings)


# ---------------------------------------------------------------------------
# The page itself
# ---------------------------------------------------------------------------


@pytest.fixture
def app_test(monkeypatch: pytest.MonkeyPatch) -> Any:
    """An ``AppTest`` over the real demo script, or a skip.

    ``AXIOM_UI_BACKEND=local`` pins in-process retrieval: an ``auto`` run would
    reach for whatever happens to be listening on 8000 on the developer's box,
    which is exactly the kind of ambient dependency TestPlan.md section 1.2
    rule 2 forbids.
    """
    pytest.importorskip("streamlit", reason="the serve extra is not installed")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("AXIOM_UI_BACKEND", "local")
    monkeypatch.delenv("AXIOM_API_BASE_URL", raising=False)
    return AppTest.from_file(str(APP_SCRIPT), default_timeout=120)


@pytest.mark.integration
@pytest.mark.slow
class TestThePageRenders:
    """TC-072, executed rather than asserted about."""

    def test_an_empty_index_shows_remediation_and_never_a_traceback(
        self, app_test: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("AXIOM_INDEX_ROOT", str(tmp_path / ".axiom"))
        app_test.run()
        assert app_test.exception == []
        assert any("no version registered" in w.value for w in app_test.warning)
        assert any("axiom index" in m.value for m in app_test.main.markdown)

    def test_a_first_load_against_a_real_index_renders_the_form(
        self, app_test: Any, indexed_v1: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings, _report = indexed_v1
        monkeypatch.setenv("AXIOM_INDEX_ROOT", str(settings.index_root))
        app_test.run()
        assert app_test.exception == []
        assert app_test.title[0].value.startswith("Axiom")
        assert [button.label for button in app_test.button] == ["Search"]
        assert any("in-process" in m.value for m in app_test.sidebar.markdown)

    def test_a_real_query_renders_one_code_block_per_result(
        self, app_test: Any, indexed_v1: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The claim is "it retrieves and shows code"; this is that claim, executed."""
        settings, _report = indexed_v1
        monkeypatch.setenv("AXIOM_INDEX_ROOT", str(settings.index_root))
        app_test.run()
        app_test.text_input[0].set_value("Which files call preprocessInput before resolveTool?")
        app_test.button[0].click().run()

        assert app_test.exception == []
        assert app_test.error == []
        assert app_test.code, "a query with results rendered no snippet"
        texts = {block.value for block in app_test.code}
        assert any("preprocessInput" in text or "resolveTool" in text for text in texts)

    def test_a_blank_query_is_refused_in_the_page_not_by_a_traceback(
        self, app_test: Any, indexed_v1: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings, _report = indexed_v1
        monkeypatch.setenv("AXIOM_INDEX_ROOT", str(settings.index_root))
        app_test.run()
        app_test.text_input[0].set_value("   ")
        app_test.button[0].click().run()
        assert app_test.exception == []
        assert any("non-whitespace" in e.value for e in app_test.error)

    def test_a_query_that_finds_nothing_says_so_instead_of_rendering_blank(
        self, app_test: Any, indexed_v1: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings, _report = indexed_v1
        monkeypatch.setenv("AXIOM_INDEX_ROOT", str(settings.index_root))
        app_test.run()
        app_test.text_input[0].set_value("中文的查询 zzqqxx")
        app_test.button[0].click().run()
        assert app_test.exception == []
        # Either results or an explicit empty state -- never a blank page.
        assert app_test.code or app_test.warning

    def test_the_sidebar_offers_every_version_and_the_type_override(
        self, app_test: Any, indexed_v1: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings, _report = indexed_v1
        monkeypatch.setenv("AXIOM_INDEX_ROOT", str(settings.index_root))
        app_test.run()
        assert app_test.exception == []
        labels = [box.label for box in app_test.sidebar.selectbox]
        assert "Retrieval path" in labels
        assert "Query type" in labels
        query_type_box = next(b for b in app_test.sidebar.selectbox if b.label == "Query type")
        assert query_type_box.options == [
            "auto",
            *[member.value for member in QueryType],
        ]
