"""Export and benchmark the KAN-11 Random Forest classifier with ONNX Runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import fmean
from time import perf_counter_ns
from typing import Callable, Final, cast

import numpy as np
import onnxruntime as ort
import polars as pl
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from techchallenge.baseline_nlp import (
    BaselineNlpConfig,
    BaselineSelection,
    DataProvenance,
    EvaluationResult,
    ExperimentResult,
    ModelingSplits,
    SplitData,
    baseline_selection,
    evaluate_classifier,
    read_dvc_provenance,
    run_and_log_experiment,
    split_modeling_base,
    train_tfidf_random_forest,
)
from techchallenge.tracking import (
    AggregateTrackingRun,
    JsonValue,
    MlflowClient,
    log_aggregate_run,
)

DEFAULT_EXPERIMENT_NAME_ONNX: Final = "kan-10-onnx-benchmark"
RANDOM_FOREST_MODEL_NAME: Final = "tfidf_random_forest"


@dataclass(frozen=True)
class OnnxBenchmarkConfig:
    """Parameters for deterministic, in-process benchmark measurements."""

    benchmark_records: int = 64
    warmup_rounds: int = 1
    repetitions: int = 5
    desired_speedup: float = 1.0
    pruning_ccp_alphas: tuple[float, ...] = (0.0001, 0.001, 0.01)


@dataclass(frozen=True)
class LatencyBenchmark:
    """Aggregate per-text latency for equivalent sklearn and ONNX flows."""

    records: int
    repetitions: int
    sklearn_mean_latency_ms: float
    onnx_mean_latency_ms: float

    @property
    def speedup(self) -> float:
        """Return the sklearn-to-ONNX mean-latency ratio."""
        return self.sklearn_mean_latency_ms / self.onnx_mean_latency_ms


@dataclass(frozen=True)
class PredictionParity:
    """Aggregate parity result retaining neither input text nor predictions."""

    records: int
    matching_predictions: int

    @property
    def parity_rate(self) -> float:
        """Return the fraction of matching classes."""
        return self.matching_predictions / self.records


@dataclass(frozen=True)
class OnnxBenchmarkResult:
    """Selection, parity, and final benchmark outcomes for KAN-10."""

    baseline_result: ExperimentResult | None
    initial_benchmark: LatencyBenchmark
    final_benchmark: LatencyBenchmark
    selected_ccp_alpha: float
    pruning_attempted: bool
    onnx_gate_met: bool
    test_prediction_parity: PredictionParity


@dataclass(frozen=True)
class OnnxBenchmarkApproval:
    """Aggregate KAN-10 outcome safe to pass to the registry stage."""

    final_speedup: float
    onnx_gate_met: bool
    selected_ccp_alpha: float
    test_prediction_parity: float


@dataclass(frozen=True)
class _PruningCandidate:
    """A validation-only candidate used if the initial ONNX gate fails."""

    config: BaselineNlpConfig
    evaluation: EvaluationResult
    benchmark: LatencyBenchmark


class OnnxTextClassifier:
    """Keep TF-IDF in process and run only the Random Forest in ONNX Runtime."""

    def __init__(self, pipeline: Pipeline) -> None:
        vectorizer, _ = pipeline_components(pipeline)
        self._vectorizer = vectorizer
        self._session = ort.InferenceSession(
            serialize_onnx_classifier(pipeline), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    def predict(self, texts: tuple[str, ...]) -> tuple[str, ...]:
        """Return classes without persisting text, vectors, or predictions."""
        if not texts:
            return ()
        vectors = self._vectorizer.transform(texts).toarray().astype(np.float32)
        output = self._session.run(None, {self._input_name: vectors})
        return tuple(str(label) for label in output[0])

    def predict_one(self, text: str) -> str:
        """Run one complete text-to-class inference path."""
        return self.predict((text,))[0]


def pipeline_components(
    pipeline: Pipeline,
) -> tuple[TfidfVectorizer, RandomForestClassifier]:
    """Return the approved KAN-10 TF-IDF and Random Forest components."""
    vectorizer = pipeline.named_steps.get("tfidf")
    classifier = pipeline.named_steps.get("random_forest")
    if not isinstance(vectorizer, TfidfVectorizer):
        raise ValueError("Pipeline tfidf step must be a TfidfVectorizer")
    if not isinstance(classifier, RandomForestClassifier):
        raise ValueError("Pipeline random_forest step must be a RandomForestClassifier")
    return vectorizer, classifier


def serialize_onnx_classifier(pipeline: Pipeline) -> bytes:
    """Serialize the approved KAN-10 Random Forest to an ONNX binary."""
    vectorizer, classifier = pipeline_components(pipeline)
    feature_count = len(vectorizer.get_feature_names_out())
    onnx_model = convert_sklearn(
        classifier,
        initial_types=[("features", FloatTensorType([None, feature_count]))],
        options={id(classifier): {"zipmap": False}},
    )
    return cast(bytes, onnx_model.SerializeToString())


def run_and_log_onnx_benchmark(
    modeling_base: pl.DataFrame,
    *,
    baseline_config: BaselineNlpConfig,
    benchmark_config: OnnxBenchmarkConfig,
    dvc_pointer_path: Path,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME_ONNX,
    tracking_uri: str | None = None,
    tracker: MlflowClient | None = None,
    prior_selection: BaselineSelection | None = None,
) -> OnnxBenchmarkResult:
    """Select on validation and check final ONNX class parity on test only."""
    _validate_benchmark_config(benchmark_config)
    baseline_result: ExperimentResult | None = None
    selection = prior_selection
    if selection is None:
        baseline_result = run_and_log_experiment(
            modeling_base,
            config=baseline_config,
            dvc_pointer_path=dvc_pointer_path,
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            tracker=tracker,
        )
        selection = baseline_selection(baseline_result)
    if selection.selected_model_name != RANDOM_FOREST_MODEL_NAME:
        raise ValueError("KAN-11 did not select the Random Forest candidate")

    splits = split_modeling_base(modeling_base)
    provenance = read_dvc_provenance(dvc_pointer_path)
    initial_model = train_tfidf_random_forest(splits.train, baseline_config)
    initial_benchmark = _benchmark_on_validation(
        initial_model, splits.validation, benchmark_config
    )
    _log_benchmark_run(
        run_name="onnx_conversion_benchmark",
        phase="onnx_conversion_benchmark",
        provenance=provenance,
        baseline_config=baseline_config,
        benchmark_config=benchmark_config,
        benchmark=initial_benchmark,
        validation_macro_f1=selection.random_forest_validation_macro_f1,
        selected_ccp_alpha=baseline_config.random_forest_ccp_alpha,
        tracking_uri=tracking_uri,
        tracker=tracker,
        experiment_name=experiment_name,
    )

    selected_config = baseline_config
    pruning_attempted = initial_benchmark.speedup < benchmark_config.desired_speedup
    onnx_gate_met = not pruning_attempted
    if pruning_attempted:
        selected_candidate = _select_pruning_candidate(
            splits=splits,
            baseline_config=baseline_config,
            benchmark_config=benchmark_config,
            baseline_validation_macro_f1=selection.random_forest_validation_macro_f1,
            provenance=provenance,
            tracking_uri=tracking_uri,
            tracker=tracker,
            experiment_name=experiment_name,
        )
        if selected_candidate is not None:
            selected_config = selected_candidate.config
            onnx_gate_met = True

    final_model = train_tfidf_random_forest(
        _combine_train_and_validation(splits), selected_config
    )
    final_onnx_model = OnnxTextClassifier(final_model)
    test_prediction_parity = compare_prediction_parity(
        final_model, final_onnx_model, splits.test
    )
    _log_parity_run(
        provenance=provenance,
        baseline_config=selected_config,
        benchmark_config=benchmark_config,
        parity=test_prediction_parity,
        tracking_uri=tracking_uri,
        tracker=tracker,
        experiment_name=experiment_name,
    )
    final_benchmark = _benchmark_on_validation(
        final_model,
        splits.validation,
        benchmark_config,
        onnx_model=final_onnx_model,
    )
    _log_benchmark_run(
        run_name="final_onnx_benchmark",
        phase="final_onnx_benchmark",
        provenance=provenance,
        baseline_config=selected_config,
        benchmark_config=benchmark_config,
        benchmark=final_benchmark,
        validation_macro_f1=None,
        selected_ccp_alpha=selected_config.random_forest_ccp_alpha,
        tracking_uri=tracking_uri,
        tracker=tracker,
        experiment_name=experiment_name,
    )
    return OnnxBenchmarkResult(
        baseline_result=baseline_result,
        initial_benchmark=initial_benchmark,
        final_benchmark=final_benchmark,
        selected_ccp_alpha=selected_config.random_forest_ccp_alpha,
        pruning_attempted=pruning_attempted,
        onnx_gate_met=onnx_gate_met,
        test_prediction_parity=test_prediction_parity,
    )


def benchmark_approval(result: OnnxBenchmarkResult) -> OnnxBenchmarkApproval:
    """Extract only the gate results required by the KAN-13 stage."""
    return OnnxBenchmarkApproval(
        final_speedup=result.final_benchmark.speedup,
        onnx_gate_met=result.onnx_gate_met,
        selected_ccp_alpha=result.selected_ccp_alpha,
        test_prediction_parity=result.test_prediction_parity.parity_rate,
    )


def compare_prediction_parity(
    sklearn_model: Pipeline, onnx_model: OnnxTextClassifier, test_data: SplitData
) -> PredictionParity:
    """Compare only predicted classes on the reserved test partition."""
    sklearn_predictions = tuple(
        str(prediction) for prediction in sklearn_model.predict(test_data.texts)
    )
    onnx_predictions = onnx_model.predict(test_data.texts)
    matches = sum(
        sklearn_prediction == onnx_prediction
        for sklearn_prediction, onnx_prediction in zip(
            sklearn_predictions, onnx_predictions, strict=True
        )
    )
    return PredictionParity(records=test_data.records, matching_predictions=matches)


def _benchmark_on_validation(
    sklearn_model: Pipeline,
    validation_data: SplitData,
    config: OnnxBenchmarkConfig,
    *,
    onnx_model: OnnxTextClassifier | None = None,
) -> LatencyBenchmark:
    references = _deterministic_references(validation_data, config.benchmark_records)
    runtime_model = (
        onnx_model if onnx_model is not None else OnnxTextClassifier(sklearn_model)
    )
    _warm_up(sklearn_model, runtime_model, references, config.warmup_rounds)
    sklearn_latencies = _measure_per_text_latency(
        lambda text: str(sklearn_model.predict((text,))[0]),
        references,
        config.repetitions,
    )
    onnx_latencies = _measure_per_text_latency(
        runtime_model.predict_one, references, config.repetitions
    )
    return LatencyBenchmark(
        records=len(references),
        repetitions=config.repetitions,
        sklearn_mean_latency_ms=fmean(sklearn_latencies),
        onnx_mean_latency_ms=fmean(onnx_latencies),
    )


def _select_pruning_candidate(
    *,
    splits: ModelingSplits,
    baseline_config: BaselineNlpConfig,
    benchmark_config: OnnxBenchmarkConfig,
    baseline_validation_macro_f1: float,
    provenance: DataProvenance,
    tracking_uri: str | None,
    tracker: MlflowClient | None,
    experiment_name: str,
) -> _PruningCandidate | None:
    acceptable_candidates: list[_PruningCandidate] = []
    for ccp_alpha in benchmark_config.pruning_ccp_alphas:
        candidate_config = replace(baseline_config, random_forest_ccp_alpha=ccp_alpha)
        candidate_model = train_tfidf_random_forest(splits.train, candidate_config)
        evaluation = evaluate_classifier(
            candidate_model,
            splits.validation,
            labels=splits.labels,
            model_name=RANDOM_FOREST_MODEL_NAME,
            split_name="validation",
        )
        benchmark = _benchmark_on_validation(
            candidate_model, splits.validation, benchmark_config
        )
        _log_benchmark_run(
            run_name=f"pruning_validation_alpha_{ccp_alpha:g}",
            phase="pruning_validation",
            provenance=provenance,
            baseline_config=candidate_config,
            benchmark_config=benchmark_config,
            benchmark=benchmark,
            validation_macro_f1=evaluation.metrics["macro_f1"],
            selected_ccp_alpha=ccp_alpha,
            tracking_uri=tracking_uri,
            tracker=tracker,
            experiment_name=experiment_name,
        )
        if (
            evaluation.metrics["macro_f1"] >= baseline_validation_macro_f1
            and benchmark.speedup >= benchmark_config.desired_speedup
        ):
            acceptable_candidates.append(
                _PruningCandidate(candidate_config, evaluation, benchmark)
            )
    if not acceptable_candidates:
        return None
    return max(
        acceptable_candidates,
        key=lambda candidate: (
            candidate.benchmark.speedup,
            -candidate.config.random_forest_ccp_alpha,
        ),
    )


def _combine_train_and_validation(splits: ModelingSplits) -> SplitData:
    return SplitData(
        texts=splits.train.texts + splits.validation.texts,
        targets=splits.train.targets + splits.validation.targets,
    )


def _deterministic_references(
    data: SplitData, requested_records: int
) -> tuple[str, ...]:
    records = min(data.records, requested_records)
    return tuple(
        data.texts[index * data.records // records] for index in range(records)
    )


def _warm_up(
    sklearn_model: Pipeline,
    onnx_model: OnnxTextClassifier,
    references: tuple[str, ...],
    warmup_rounds: int,
) -> None:
    for _ in range(warmup_rounds):
        for text in references:
            sklearn_model.predict((text,))
            onnx_model.predict_one(text)


def _measure_per_text_latency(
    predict_one: Callable[[str], str],
    references: tuple[str, ...],
    repetitions: int,
) -> tuple[float, ...]:
    latencies: list[float] = []
    for _ in range(repetitions):
        started_at = perf_counter_ns()
        for text in references:
            predict_one(text)
        elapsed_ns = perf_counter_ns() - started_at
        latencies.append(elapsed_ns / len(references) / 1_000_000)
    return tuple(latencies)


def _validate_benchmark_config(config: OnnxBenchmarkConfig) -> None:
    if config.benchmark_records < 1:
        raise ValueError("benchmark_records must be at least one")
    if config.warmup_rounds < 0:
        raise ValueError("warmup_rounds cannot be negative")
    if config.repetitions < 1:
        raise ValueError("repetitions must be at least one")
    if config.desired_speedup <= 0:
        raise ValueError("desired_speedup must be positive")
    if any(alpha <= 0 for alpha in config.pruning_ccp_alphas):
        raise ValueError("pruning_ccp_alphas must contain only positive values")


def _log_benchmark_run(
    *,
    run_name: str,
    phase: str,
    provenance: DataProvenance,
    baseline_config: BaselineNlpConfig,
    benchmark_config: OnnxBenchmarkConfig,
    benchmark: LatencyBenchmark,
    validation_macro_f1: float | None,
    selected_ccp_alpha: float,
    tracking_uri: str | None,
    tracker: MlflowClient | None,
    experiment_name: str,
) -> None:
    metrics = {
        "benchmark.sklearn_mean_latency_ms": benchmark.sklearn_mean_latency_ms,
        "benchmark.onnx_mean_latency_ms": benchmark.onnx_mean_latency_ms,
        "benchmark.speedup": benchmark.speedup,
    }
    if validation_macro_f1 is not None:
        metrics["validation.macro_f1"] = validation_macro_f1
    artifact: dict[str, JsonValue] = {
        "benchmark": {
            "records": benchmark.records,
            "repetitions": benchmark.repetitions,
            "sklearn_mean_latency_ms": benchmark.sklearn_mean_latency_ms,
            "onnx_mean_latency_ms": benchmark.onnx_mean_latency_ms,
            "speedup": benchmark.speedup,
        },
        "data_provenance": {
            "dvc_pointer": provenance.pointer_path,
            "dvc_md5": provenance.md5,
        },
        "privacy": _privacy_artifact(),
    }
    if validation_macro_f1 is not None:
        artifact["validation"] = {
            "macro_f1": validation_macro_f1,
        }
    log_aggregate_run(
        AggregateTrackingRun(
            run_name=run_name,
            parameters=_tracking_parameters(
                provenance,
                baseline_config,
                benchmark_config,
                phase,
                selected_ccp_alpha,
            ),
            metrics=metrics,
            artifact=artifact,
        ),
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        client=tracker,
    )


def _log_parity_run(
    *,
    provenance: DataProvenance,
    baseline_config: BaselineNlpConfig,
    benchmark_config: OnnxBenchmarkConfig,
    parity: PredictionParity,
    tracking_uri: str | None,
    tracker: MlflowClient | None,
    experiment_name: str,
) -> None:
    log_aggregate_run(
        AggregateTrackingRun(
            run_name="final_test_prediction_parity",
            parameters=_tracking_parameters(
                provenance,
                baseline_config,
                benchmark_config,
                "final_test_prediction_parity",
                baseline_config.random_forest_ccp_alpha,
            ),
            metrics={
                "test.prediction_parity": parity.parity_rate,
                "test.matching_predictions": float(parity.matching_predictions),
            },
            artifact={
                "data_provenance": {
                    "dvc_pointer": provenance.pointer_path,
                    "dvc_md5": provenance.md5,
                },
                "prediction_parity": {
                    "records": parity.records,
                    "matching_predictions": parity.matching_predictions,
                    "parity_rate": parity.parity_rate,
                },
                "privacy": _privacy_artifact(),
            },
        ),
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        client=tracker,
    )


def _tracking_parameters(
    provenance: DataProvenance,
    baseline_config: BaselineNlpConfig,
    benchmark_config: OnnxBenchmarkConfig,
    phase: str,
    selected_ccp_alpha: float,
) -> dict[str, str]:
    return {
        "data.dvc_pointer": provenance.pointer_path,
        "data.dvc_md5": provenance.md5,
        "model.name": RANDOM_FOREST_MODEL_NAME,
        "model.ccp_alpha": str(selected_ccp_alpha),
        "run.phase": phase,
        **{
            f"baseline_config.{key}": str(value)
            for key, value in asdict(baseline_config).items()
        },
        **{
            f"benchmark_config.{key}": str(value)
            for key, value in asdict(benchmark_config).items()
        },
    }


def _privacy_artifact() -> dict[str, JsonValue]:
    return {
        "contains_model_artifact": False,
        "contains_record_identifiers": False,
        "contains_text": False,
    }
