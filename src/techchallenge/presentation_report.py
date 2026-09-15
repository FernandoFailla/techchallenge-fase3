"""Render safe aggregate evidence for the Tech Challenge presentation."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, Protocol

from mlflow import MlflowClient
from mlflow.entities import Run
from mlflow.exceptions import MlflowException

from techchallenge.baseline_nlp import DEFAULT_EXPERIMENT_NAME as BASELINE_EXPERIMENT
from techchallenge.http_benchmark import DEFAULT_EXPERIMENT_NAME_HTTP
from techchallenge.model_registry import CHAMPION_ALIAS, MODEL_NAME
from techchallenge.onnx_benchmark import DEFAULT_EXPERIMENT_NAME_ONNX
from techchallenge.tracking import get_tracking_uri

EXPECTED_WARMUPS: Final = "20"
EXPECTED_MEASUREMENTS: Final = "500"
SLIDES_TEMPLATE: Final = Path("docs/apresentacao-slides.md")


@dataclass(frozen=True)
class TrackedRun:
    """Safe MLflow run fields used by the report."""

    run_id: str
    started_at_ms: int
    metrics: dict[str, float]
    parameters: dict[str, str]
    tags: dict[str, str]


@dataclass(frozen=True)
class ChampionSnapshot:
    """Current registered model alias without model artifact contents."""

    version: str
    run_id: str
    status: str


@dataclass(frozen=True)
class PresentationEvidence:
    """Latest aggregate observations required by the video."""

    commit: str
    generated_at: datetime
    tracking_uri: str
    baseline: TrackedRun | None
    onnx_benchmark: TrackedRun | None
    onnx_parity: TrackedRun | None
    http_benchmark: TrackedRun | None
    champion: ChampionSnapshot | None
    champion_run: TrackedRun | None


class PresentationReader(Protocol):
    """Read-only tracking contract used by collection and test doubles."""

    def finished_runs(self, experiment_name: str) -> tuple[TrackedRun, ...]: ...

    def champion(self) -> ChampionSnapshot | None: ...

    def run(self, run_id: str) -> TrackedRun: ...


class MlflowPresentationReader:
    """Read aggregate run metadata from the configured MLflow server."""

    def __init__(self, tracking_uri: str) -> None:
        self._client = MlflowClient(tracking_uri=tracking_uri)

    def finished_runs(self, experiment_name: str) -> tuple[TrackedRun, ...]:
        """Return recent completed runs in descending start order."""
        experiment = self._client.get_experiment_by_name(experiment_name)
        if experiment is None:
            return ()
        runs = self._client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=["attributes.start_time DESC"],
            max_results=1000,
        )
        return tuple(_tracked_run(run) for run in runs)

    def champion(self) -> ChampionSnapshot | None:
        """Resolve the current champion alias when it exists."""
        try:
            version = self._client.get_model_version_by_alias(
                MODEL_NAME, CHAMPION_ALIAS
            )
        except MlflowException:
            return None
        if version.run_id is None:
            return None
        return ChampionSnapshot(
            version=str(version.version),
            run_id=version.run_id,
            status=str(version.status),
        )

    def run(self, run_id: str) -> TrackedRun:
        """Read one run by its immutable identifier."""
        return _tracked_run(self._client.get_run(run_id))


def collect_presentation_evidence(
    reader: PresentationReader,
    *,
    commit: str,
    generated_at: datetime,
    tracking_uri: str,
) -> PresentationEvidence:
    """Select the latest relevant completed run for every evidence category."""
    baseline = _latest_by_parameter(
        reader.finished_runs(BASELINE_EXPERIMENT), "run.phase", "final_test_report"
    )
    onnx_runs = reader.finished_runs(DEFAULT_EXPERIMENT_NAME_ONNX)
    onnx_benchmark = _latest_by_parameter(
        onnx_runs, "run.phase", "final_onnx_benchmark"
    )
    onnx_parity = _latest_by_parameter(
        onnx_runs, "run.phase", "final_test_prediction_parity"
    )
    http_benchmark = _latest_by_tag(
        reader.finished_runs(DEFAULT_EXPERIMENT_NAME_HTTP),
        "mlflow.runName",
        "sequential_http_benchmark",
    )
    champion = reader.champion()
    champion_run = reader.run(champion.run_id) if champion is not None else None
    return PresentationEvidence(
        commit=commit,
        generated_at=generated_at,
        tracking_uri=tracking_uri,
        baseline=baseline,
        onnx_benchmark=onnx_benchmark,
        onnx_parity=onnx_parity,
        http_benchmark=http_benchmark,
        champion=champion,
        champion_run=champion_run,
    )


def render_presentation_report(evidence: PresentationEvidence) -> str:
    """Render a copy-ready Markdown report containing aggregates only."""
    baseline = evidence.baseline
    onnx = evidence.onnx_benchmark
    parity = evidence.onnx_parity
    http = evidence.http_benchmark
    champion = evidence.champion
    champion_run = evidence.champion_run
    measured_predictions = _metric(onnx, "benchmark.measured_predictions", integer=True)
    matching_predictions = _metric(parity, "test.matching_predictions", integer=True)
    test_records = _parity_records(parity)
    champion_parity = _metric(champion_run, "kan10.test_prediction_parity")
    champion_speedup = _metric(champion_run, "kan10.onnx_speedup")
    measured_requests = _metric(http, "http.measured_requests", integer=True)
    lines = [
        "# Evidencias Para A Apresentacao",
        "",
        f"- Gerado em UTC: `{evidence.generated_at.astimezone(UTC).isoformat()}`",
        f"- Commit: `{evidence.commit}`",
        "",
        "## Qualidade Do Modelo",
        "",
        "| Evidencia | Valor |",
        "|---|---:|",
        f"| Modelo selecionado | {_parameter(baseline, 'model.name')} |",
        f"| Registros avaliados no teste | {test_records} |",
        f"| Acuracia | {_metric(baseline, 'test.accuracy')} |",
        f"| Macro-F1 | {_metric(baseline, 'test.macro_f1')} |",
        f"| Precisao macro | {_metric(baseline, 'test.macro_precision')} |",
        f"| Recall macro | {_metric(baseline, 'test.macro_recall')} |",
        "",
        "## Comparacao De Latencia",
        "",
        "| Metrica | scikit-learn (ms) | ONNX (ms) | Reducao | Speedup |",
        "|---|---:|---:|---:|---:|",
        _latency_row(onnx, "Media", "mean"),
        _latency_row(onnx, "P50", "p50"),
        _latency_row(onnx, "P95", "p95"),
        "",
        f"- Warm-ups: {_parameter(onnx, 'benchmark_config.warmup_predictions')}.",
        f"- Predicoes medidas: {measured_predictions}.",
        f"- Paridade no teste: {_metric(parity, 'test.prediction_parity')}.",
        f"- Predicoes coincidentes: {matching_predictions}.",
        "",
        "## Modelo Servido E API",
        "",
        "| Evidencia | Valor |",
        "|---|---:|",
        f"| Modelo registrado | `{MODEL_NAME}` |",
        f"| Alias | `{CHAMPION_ALIAS}` |",
        f"| Versao champion | {_champion_value(champion, 'version')} |",
        f"| Status champion | {_champion_value(champion, 'status')} |",
        f"| Paridade registrada no champion | {champion_parity} |",
        f"| Speedup registrado no champion | {champion_speedup} |",
        f"| Versao medida pela API | {_parameter(http, 'model.version')} |",
        f"| Requisicoes HTTP medidas | {measured_requests} |",
        f"| HTTP media | {_metric(http, 'http.mean_latency_ms', suffix=' ms')} |",
        f"| HTTP P50 | {_metric(http, 'http.p50_latency_ms', suffix=' ms')} |",
        f"| HTTP P95 | {_metric(http, 'http.p95_latency_ms', suffix=' ms')} |",
        "",
        "## Verificacoes",
        "",
        *_checks(evidence),
        "",
        "## Limites Da Evidencia",
        "",
        "- As latencias sao medicoes locais e dependem da maquina e da carga.",
        (
            "- O benchmark ONNX compara o caminho texto-ate-classe; o benchmark "
            "HTTP mede somente o champion servido."
        ),
        (
            "- As runs mais recentes sao selecionadas por experimento. O hash DVC "
            "detecta parte das inconsistencias, mas nao substitui um ID comum da DAG."
        ),
        (
            "- O MLflow atual nao registra o commit Git da DAG; o commit acima "
            "identifica apenas o codigo usado para gerar este relatorio."
        ),
        (
            "- O relatorio nao le textos, predicoes individuais, vetores TF-IDF ou "
            "artefatos do modelo."
        ),
    ]
    return "\n".join(lines) + "\n"


def render_presentation_slides(evidence: PresentationEvidence, template: str) -> str:
    """Fill the versioned slide template with the latest aggregate evidence."""
    onnx = evidence.onnx_benchmark
    parity = evidence.onnx_parity
    baseline = evidence.baseline
    http = evidence.http_benchmark
    generated_at = evidence.generated_at.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
    metadata = (
        f"Commit `{evidence.commit}` | Champion v"
        f"{_champion_value(evidence.champion, 'version')} | {generated_at}"
    )
    quality = (
        f"**Macro-F1 {_metric(baseline, 'test.macro_f1')} | "
        f"Acuracia {_metric(baseline, 'test.accuracy')} | "
        f"Teste N={_parity_records(parity)}**"
    )
    onnx_results = "\n".join(
        (
            "| Latencia | scikit-learn | ONNX |",
            "|---|---:|---:|",
            _slide_latency_row(onnx, "Media", "mean"),
            _slide_latency_row(onnx, "P50", "p50"),
            _slide_latency_row(onnx, "P95", "p95"),
            "",
            f"**Speedup {_metric(onnx, 'benchmark.speedup', suffix='x')} | "
            f"Paridade {_metric(parity, 'test.prediction_parity')}**",
            "",
            f"{_parameter(onnx, 'benchmark_config.warmup_predictions')} warm-ups | "
            f"{_metric(onnx, 'benchmark.measured_predictions', integer=True)} "
            "predicoes",
        )
    )
    result = "\n".join(
        (
            "| Evidencia | Resultado local |",
            "|---|---:|",
            f"| Champion | v{_champion_value(evidence.champion, 'version')} |",
            f"| Macro-F1 | {_metric(baseline, 'test.macro_f1')} |",
            f"| ONNX speedup | {_metric(onnx, 'benchmark.speedup', suffix='x')} |",
            f"| HTTP P50 | {_metric(http, 'http.p50_latency_ms', suffix=' ms')} |",
            f"| HTTP P95 | {_metric(http, 'http.p95_latency_ms', suffix=' ms')} |",
            "| HTTP requests | "
            f"{_metric(http, 'http.measured_requests', integer=True)} |",
        )
    )
    replacements = {
        "<!-- PRESENTATION_METADATA -->": metadata,
        "<!-- PRESENTATION_QUALITY -->": quality,
        "<!-- PRESENTATION_ONNX_RESULTS -->": onnx_results,
        "<!-- PRESENTATION_RESULT -->": result,
    }
    rendered = template
    for marker, value in replacements.items():
        rendered = rendered.replace(marker, value)
    return rendered


def _tracked_run(run: Run) -> TrackedRun:
    return TrackedRun(
        run_id=run.info.run_id,
        started_at_ms=run.info.start_time or 0,
        metrics=dict(run.data.metrics),
        parameters=dict(run.data.params),
        tags=dict(run.data.tags),
    )


def _latest_by_parameter(
    runs: tuple[TrackedRun, ...], key: str, value: str
) -> TrackedRun | None:
    matches = tuple(run for run in runs if run.parameters.get(key) == value)
    return max(matches, key=lambda run: run.started_at_ms, default=None)


def _latest_by_tag(
    runs: tuple[TrackedRun, ...], key: str, value: str
) -> TrackedRun | None:
    matches = tuple(run for run in runs if run.tags.get(key) == value)
    return max(matches, key=lambda run: run.started_at_ms, default=None)


def _metric(
    run: TrackedRun | None,
    key: str,
    *,
    integer: bool = False,
    suffix: str = "",
) -> str:
    if run is None or key not in run.metrics:
        return "nao registrado"
    value = run.metrics[key]
    formatted = str(round(value)) if integer else f"{value:.3f}"
    return f"{formatted}{suffix}"


def _parameter(run: TrackedRun | None, key: str) -> str:
    if run is None:
        return "nao registrado"
    return run.parameters.get(key, "nao registrado")


def _parity_records(run: TrackedRun | None) -> str:
    matching = _raw_metric(run, "test.matching_predictions")
    parity = _raw_metric(run, "test.prediction_parity")
    if matching is None or parity is None or parity <= 0:
        return "nao registrado"
    return str(round(matching / parity))


def _champion_value(
    champion: ChampionSnapshot | None, field: Literal["version", "status"]
) -> str:
    if champion is None:
        return "nao registrado"
    return champion.version if field == "version" else champion.status


def _latency_row(run: TrackedRun | None, label: str, metric: str) -> str:
    sklearn = _raw_metric(run, f"benchmark.sklearn_{metric}_latency_ms")
    onnx = _raw_metric(run, f"benchmark.onnx_{metric}_latency_ms")
    if sklearn is None or onnx is None or sklearn <= 0 or onnx <= 0:
        return (
            f"| {label} | nao registrado | nao registrado | nao registrado | "
            "nao registrado |"
        )
    reduction = (1 - onnx / sklearn) * 100
    speedup = sklearn / onnx
    return (
        f"| {label} | {sklearn:.3f} | {onnx:.3f} | {reduction:.1f}% | {speedup:.3f}x |"
    )


def _slide_latency_row(run: TrackedRun | None, label: str, metric: str) -> str:
    sklearn = _metric(run, f"benchmark.sklearn_{metric}_latency_ms", suffix=" ms")
    onnx = _metric(run, f"benchmark.onnx_{metric}_latency_ms", suffix=" ms")
    return f"| {label} | {sklearn} | {onnx} |"


def _raw_metric(run: TrackedRun | None, key: str) -> float | None:
    if run is None:
        return None
    return run.metrics.get(key)


def presentation_is_ready(evidence: PresentationEvidence) -> bool:
    """Return whether every required consistency check passes."""
    return all(passed for _, passed in _check_results(evidence))


def _checks(evidence: PresentationEvidence) -> list[str]:
    return [_check(label, passed) for label, passed in _check_results(evidence)]


def _check_results(evidence: PresentationEvidence) -> tuple[tuple[str, bool], ...]:
    onnx = evidence.onnx_benchmark
    parity = evidence.onnx_parity
    http = evidence.http_benchmark
    return (
        (
            "Codigo demonstrado sem alteracoes locais",
            not evidence.commit.endswith("-dirty"),
        ),
        (
            "Protocolo ONNX 20 warm-ups / 500 medicoes",
            onnx is not None
            and onnx.parameters.get("benchmark_config.warmup_predictions")
            == EXPECTED_WARMUPS
            and onnx.metrics.get("benchmark.measured_predictions") == 500.0,
        ),
        (
            "Paridade ONNX de 100%",
            parity is not None and parity.metrics.get("test.prediction_parity") == 1.0,
        ),
        (
            "Ganho medio ONNX positivo",
            onnx is not None and onnx.metrics.get("benchmark.speedup", 0.0) > 1.0,
        ),
        (
            "Versao HTTP igual ao champion",
            evidence.champion is not None
            and http is not None
            and http.parameters.get("model.version") == evidence.champion.version,
        ),
        ("Mesmo hash DVC nas runs", _matching_dvc_hashes(evidence)),
        (
            "Configuracao avaliada igual ao champion",
            _matching_model_configurations(evidence),
        ),
    )


def _matching_dvc_hashes(evidence: PresentationEvidence) -> bool:
    runs = tuple(
        run
        for run in (
            evidence.baseline,
            evidence.onnx_benchmark,
            evidence.onnx_parity,
            evidence.champion_run,
        )
        if run is not None
    )
    hashes = tuple(run.parameters.get("data.dvc_md5") for run in runs)
    return (
        len(runs) == 4
        and all(isinstance(value, str) and bool(value) for value in hashes)
        and len(set(hashes)) == 1
    )


def _matching_model_configurations(evidence: PresentationEvidence) -> bool:
    if (
        evidence.baseline is None
        or evidence.onnx_benchmark is None
        or evidence.onnx_parity is None
        or evidence.champion_run is None
    ):
        return False
    baseline_config = _parameter_group(evidence.baseline, "config.")
    onnx_config = _parameter_group(evidence.onnx_benchmark, "baseline_config.")
    parity_config = _parameter_group(evidence.onnx_parity, "baseline_config.")
    champion_config = _parameter_group(evidence.champion_run, "baseline_config.")
    return (
        bool(baseline_config)
        and baseline_config == onnx_config == parity_config == champion_config
    )


def _parameter_group(run: TrackedRun, prefix: str) -> dict[str, str]:
    return {
        key.removeprefix(prefix): value
        for key, value in run.parameters.items()
        if key.startswith(prefix)
    }


def _check(label: str, passed: bool) -> str:
    return f"- [{'OK' if passed else 'PENDENTE'}] {label}."


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            ".",
            ":(exclude).ai-jail",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    suffix = "-dirty" if status.stdout.strip() else ""
    return f"{result.stdout.strip()}{suffix}"


def main() -> None:
    """Print the latest presentation evidence without persisting sensitive data."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slides", action="store_true", help="render the populated Marp slide deck"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit with an error when presentation checks are pending",
    )
    arguments = parser.parse_args()
    tracking_uri = get_tracking_uri()
    try:
        evidence = collect_presentation_evidence(
            MlflowPresentationReader(tracking_uri),
            commit=_git_commit(),
            generated_at=datetime.now(UTC),
            tracking_uri=tracking_uri,
        )
    except MlflowException as error:
        raise SystemExit(
            "Nao foi possivel consultar o MLflow. Inicie-o com `make mlflow` "
            "e execute a DAG training_pipeline."
        ) from error
    if arguments.strict and not presentation_is_ready(evidence):
        if not arguments.slides:
            print(render_presentation_report(evidence), end="")
        raise SystemExit(
            "As evidencias possuem verificacoes pendentes; consulte "
            "`make presentation-report`."
        )
    if arguments.slides:
        print(
            render_presentation_slides(
                evidence, SLIDES_TEMPLATE.read_text(encoding="utf-8")
            ),
            end="",
        )
    else:
        print(render_presentation_report(evidence), end="")


if __name__ == "__main__":
    main()
