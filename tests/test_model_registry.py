from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pytest
from mlflow.pyfunc.model import PythonModelContext
from sklearn.pipeline import Pipeline

from techchallenge.baseline_nlp import (
    BaselineNlpConfig,
    DataProvenance,
    SplitData,
    train_tfidf_random_forest,
)
from techchallenge.model_registry import (
    OnnxUrgencyPyfunc,
    _build_manifest,
    _bundle_hashes,
    _registry_parameters,
    _require_approved_kan10_result,
    _run_sha256,
    _write_bundle,
)
from techchallenge.onnx_benchmark import OnnxBenchmarkConfig, pipeline_components


def _trained_pipeline() -> Pipeline:
    return train_tfidf_random_forest(
        SplitData(
            texts=("amber", "azure", "amber amber", "azure azure"),
            targets=("high", "low", "high", "low"),
        ),
        BaselineNlpConfig(
            min_document_frequency=1,
            random_forest_estimators=5,
            random_forest_n_jobs=1,
        ),
    )


def test_pyfunc_bundle_predicts_with_onnx_and_does_not_expose_vocabulary(
    tmp_path: Path,
) -> None:
    pipeline = _trained_pipeline()
    vectorizer, classifier = pipeline_components(pipeline)
    paths = _write_bundle(tmp_path, vectorizer, classifier, pipeline)
    pyfunc = OnnxUrgencyPyfunc()
    context = PythonModelContext(  # type: ignore[no-untyped-call]
        artifacts={artifact_name: str(path) for artifact_name, path in paths.items()},
        model_config={},
    )

    pyfunc.load_context(context)
    predictions = pyfunc.predict(context, pd.DataFrame({"text": ["amber", "azure"]}))

    assert list(predictions.columns) == ["urgency"]
    assert len(predictions) == 2
    assert set(predictions["urgency"]).issubset({"high", "low"})
    class_mapping = json.loads(paths["class_mapping"].read_text(encoding="utf-8"))
    assert class_mapping == {"0": "high", "1": "low"}


def test_bundle_hashes_and_manifest_only_contain_aggregate_provenance(
    tmp_path: Path,
) -> None:
    pipeline = _trained_pipeline()
    vectorizer, classifier = pipeline_components(pipeline)
    paths = _write_bundle(tmp_path, vectorizer, classifier, pipeline)
    hashes = _bundle_hashes(paths)
    manifest = _build_manifest(
        provenance=DataProvenance(pointer_path="data/base.dvc", md5="a" * 32),
        baseline_config=BaselineNlpConfig(),
        benchmark_config=OnnxBenchmarkConfig(),
        hashes=hashes,
    )

    assert len(hashes.vectorizer_sha256) == 64
    assert len(hashes.onnx_model_sha256) == 64
    assert len(hashes.class_mapping_sha256) == 64
    serialized_manifest = json.dumps(manifest)
    assert "amber" not in serialized_manifest
    assert "azure" not in serialized_manifest
    assert manifest["privacy"] == {
        "contains_clinical_text": False,
        "contains_record_identifiers": False,
        "fitted_vocabulary_exposed_in_tracking_metadata": False,
        "contains_secrets": False,
    }


def test_registry_parameters_include_run_provenance_model_and_bundle_hashes(
    tmp_path: Path,
) -> None:
    pipeline = _trained_pipeline()
    vectorizer, classifier = pipeline_components(pipeline)
    hashes = _bundle_hashes(_write_bundle(tmp_path, vectorizer, classifier, pipeline))
    provenance = DataProvenance(pointer_path="data/base.dvc", md5="a" * 32)
    hashes = replace(hashes, manifest_sha256="b" * 64)
    hashes = replace(
        hashes,
        run_sha256=_run_sha256(
            provenance=provenance,
            baseline_config=BaselineNlpConfig(),
            benchmark_config=OnnxBenchmarkConfig(),
            hashes=hashes,
        ),
    )

    parameters = _registry_parameters(
        provenance=provenance,
        baseline_config=BaselineNlpConfig(),
        benchmark_config=OnnxBenchmarkConfig(),
        hashes=hashes,
    )

    assert {
        "run.integrity_sha256",
        "provenance.data_dvc_md5",
        "provenance.source_code_sha256",
        "model.vectorizer_sha256",
        "model.onnx_sha256",
        "model.class_mapping_sha256",
        "bundle.manifest_sha256",
    }.issubset(parameters)


@pytest.mark.parametrize(
    ("onnx_gate_met", "prediction_parity"),
    [(False, 1.0), (True, 0.99)],
)
def test_registry_gate_rejects_unapproved_kan10_result(
    onnx_gate_met: bool, prediction_parity: float
) -> None:
    with pytest.raises(ValueError):
        _require_approved_kan10_result(
            onnx_gate_met=onnx_gate_met, prediction_parity=prediction_parity
        )
