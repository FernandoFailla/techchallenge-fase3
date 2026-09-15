from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from techchallenge.presentation_report import (
    ChampionSnapshot,
    PresentationEvidence,
    TrackedRun,
    collect_presentation_evidence,
    presentation_is_ready,
    render_presentation_report,
    render_presentation_slides,
)


class FakePresentationReader:
    def __init__(
        self,
        runs: dict[str, tuple[TrackedRun, ...]],
        champion: ChampionSnapshot | None,
    ) -> None:
        self._runs = runs
        self._champion = champion

    def finished_runs(self, experiment_name: str) -> tuple[TrackedRun, ...]:
        return self._runs.get(experiment_name, ())

    def champion(self) -> ChampionSnapshot | None:
        return self._champion

    def run(self, run_id: str) -> TrackedRun:
        return next(
            run for runs in self._runs.values() for run in runs if run.run_id == run_id
        )


def _run(
    run_id: str,
    started_at_ms: int,
    *,
    metrics: dict[str, float] | None = None,
    parameters: dict[str, str] | None = None,
    tags: dict[str, str] | None = None,
) -> TrackedRun:
    return TrackedRun(
        run_id=run_id,
        started_at_ms=started_at_ms,
        metrics=metrics or {},
        parameters=parameters or {},
        tags=tags or {},
    )


def test_collection_selects_latest_runs_by_phase_and_run_name() -> None:
    old = _run("old", 1, parameters={"run.phase": "final_test_report"})
    latest = _run("latest", 2, parameters={"run.phase": "final_test_report"})
    onnx = _run("onnx", 3, parameters={"run.phase": "final_onnx_benchmark"})
    parity = _run("parity", 4, parameters={"run.phase": "final_test_prediction_parity"})
    http = _run("http", 5, tags={"mlflow.runName": "sequential_http_benchmark"})
    registry = _run("registry", 6)
    reader = FakePresentationReader(
        {
            "kan-11-baseline-nlp": (old, latest),
            "kan-10-onnx-benchmark": (onnx, parity),
            "kan-24-http-benchmark": (http,),
            "registry": (registry,),
        },
        ChampionSnapshot(version="7", run_id="registry", status="READY"),
    )

    evidence = collect_presentation_evidence(
        reader,
        commit="abc1234",
        generated_at=datetime(2026, 9, 14, tzinfo=UTC),
        tracking_uri="http://localhost:5000",
    )

    assert evidence.baseline == latest
    assert evidence.onnx_benchmark == onnx
    assert evidence.onnx_parity == parity
    assert evidence.http_benchmark == http
    assert evidence.champion_run == registry


def test_report_calculates_latency_gains_and_consistency_checks() -> None:
    common_parameters = {"data.dvc_md5": "a" * 32}
    baseline = _run(
        "baseline",
        1,
        metrics={"test.macro_f1": 0.758},
        parameters={
            **common_parameters,
            "model.name": "tfidf_random_forest",
            "config.random_forest_ccp_alpha": "0.0",
        },
    )
    onnx = _run(
        "onnx",
        2,
        metrics={
            "benchmark.sklearn_mean_latency_ms": 10.0,
            "benchmark.sklearn_p50_latency_ms": 8.0,
            "benchmark.sklearn_p95_latency_ms": 12.0,
            "benchmark.onnx_mean_latency_ms": 2.0,
            "benchmark.onnx_p50_latency_ms": 2.0,
            "benchmark.onnx_p95_latency_ms": 3.0,
            "benchmark.measured_predictions": 500.0,
            "benchmark.speedup": 5.0,
        },
        parameters={
            **common_parameters,
            "benchmark_config.warmup_predictions": "20",
            "baseline_config.random_forest_ccp_alpha": "0.0",
        },
    )
    parity = _run(
        "parity",
        3,
        metrics={
            "test.prediction_parity": 1.0,
            "test.matching_predictions": 299.0,
        },
        parameters={
            **common_parameters,
            "baseline_config.random_forest_ccp_alpha": "0.0",
        },
    )
    http = _run("http", 4, parameters={"model.version": "7"})
    registry = _run(
        "registry",
        5,
        parameters={
            **common_parameters,
            "baseline_config.random_forest_ccp_alpha": "0.0",
        },
    )
    evidence = PresentationEvidence(
        commit="abc1234",
        generated_at=datetime(2026, 9, 14, tzinfo=UTC),
        tracking_uri="http://localhost:5000",
        baseline=baseline,
        onnx_benchmark=onnx,
        onnx_parity=parity,
        http_benchmark=http,
        champion=ChampionSnapshot(version="7", run_id="registry", status="READY"),
        champion_run=registry,
    )

    report = render_presentation_report(evidence)

    assert "| Media | 10.000 | 2.000 | 80.0% | 5.000x |" in report
    assert "| Registros avaliados no teste | 299 |" in report
    assert "[OK] Protocolo ONNX 20 warm-ups / 500 medicoes." in report
    assert "[OK] Paridade ONNX de 100%." in report
    assert "[OK] Versao HTTP igual ao champion." in report
    assert "[OK] Mesmo hash DVC nas runs." in report
    assert "[OK] Configuracao avaliada igual ao champion." in report
    assert presentation_is_ready(evidence)

    pruned_onnx = replace(
        onnx,
        parameters={
            **onnx.parameters,
            "baseline_config.random_forest_ccp_alpha": "0.01",
        },
    )
    inconsistent = replace(evidence, onnx_benchmark=pruned_onnx)

    assert not presentation_is_ready(inconsistent)
    assert (
        "[PENDENTE] Configuracao avaliada igual ao champion."
        in render_presentation_report(inconsistent)
    )

    inconsistent_parity = replace(
        evidence,
        onnx_parity=replace(
            parity,
            parameters={
                **parity.parameters,
                "baseline_config.random_forest_ccp_alpha": "0.01",
            },
        ),
    )

    assert not presentation_is_ready(inconsistent_parity)


def test_report_marks_missing_evidence_without_using_zero() -> None:
    evidence = PresentationEvidence(
        commit="abc1234",
        generated_at=datetime(2026, 9, 14, tzinfo=UTC),
        tracking_uri="http://localhost:5000",
        baseline=None,
        onnx_benchmark=None,
        onnx_parity=None,
        http_benchmark=None,
        champion=None,
        champion_run=None,
    )

    report = render_presentation_report(evidence)

    assert "nao registrado" in report
    assert "[PENDENTE] Paridade ONNX de 100%." in report
    assert not presentation_is_ready(evidence)


def test_report_does_not_accept_missing_dvc_hashes_as_consistent() -> None:
    run_with_hash = _run("baseline", 1, parameters={"data.dvc_md5": "a" * 32})
    run_without_hash = _run("missing", 2)
    evidence = PresentationEvidence(
        commit="abc1234",
        generated_at=datetime(2026, 9, 14, tzinfo=UTC),
        tracking_uri="http://localhost:5000",
        baseline=run_with_hash,
        onnx_benchmark=run_without_hash,
        onnx_parity=run_without_hash,
        http_benchmark=None,
        champion=ChampionSnapshot(version="7", run_id="missing", status="READY"),
        champion_run=run_without_hash,
    )

    report = render_presentation_report(evidence)

    assert "[PENDENTE] Mesmo hash DVC nas runs." in report


def test_slide_template_is_filled_with_current_aggregate_values() -> None:
    evidence = PresentationEvidence(
        commit="abc1234",
        generated_at=datetime(2026, 9, 14, tzinfo=UTC),
        tracking_uri="http://localhost:5000",
        baseline=_run("baseline", 1, metrics={"test.macro_f1": 0.758}),
        onnx_benchmark=_run(
            "onnx",
            2,
            metrics={"benchmark.speedup": 5.0},
            parameters={"benchmark_config.warmup_predictions": "20"},
        ),
        onnx_parity=_run("parity", 3, metrics={"test.prediction_parity": 1.0}),
        http_benchmark=None,
        champion=ChampionSnapshot(version="7", run_id="registry", status="READY"),
        champion_run=None,
    )

    rendered = render_presentation_slides(
        evidence,
        "<!-- PRESENTATION_METADATA -->\n<!-- PRESENTATION_QUALITY -->\n"
        "<!-- PRESENTATION_ONNX_RESULTS -->\n<!-- PRESENTATION_RESULT -->",
    )

    assert "Commit `abc1234` | Champion v7" in rendered
    assert "Macro-F1 0.758" in rendered
    assert "Speedup 5.000x" in rendered
    assert "| Champion | v7 |" in rendered


def test_versioned_slide_template_has_no_unknown_markers() -> None:
    evidence = PresentationEvidence(
        commit="abc1234",
        generated_at=datetime(2026, 9, 14, tzinfo=UTC),
        tracking_uri="http://localhost:5000",
        baseline=None,
        onnx_benchmark=None,
        onnx_parity=None,
        http_benchmark=None,
        champion=None,
        champion_run=None,
    )
    template = Path("docs/apresentacao-slides.md").read_text(encoding="utf-8")

    rendered = render_presentation_slides(evidence, template)

    assert "<!-- PRESENTATION_" not in rendered
