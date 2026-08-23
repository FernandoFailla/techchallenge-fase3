from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import mlflow
import polars as pl
from pytest import MonkeyPatch

from techchallenge import model_registry
from techchallenge.baseline_nlp import BaselineNlpConfig
from techchallenge.model_registry import RegistryConfig, register_approved_onnx_model
from techchallenge.onnx_benchmark import OnnxBenchmarkApproval, OnnxBenchmarkConfig
from techchallenge.training_pipeline import (
    ModelingBaseReference,
    OptimizationReference,
    RegistrationReference,
    TrainingReference,
)


def test_xcom_handoffs_only_contain_paths_run_ids_and_aggregate_metadata() -> None:
    modeling_base = ModelingBaseReference(
        dvc_md5="a" * 32,
        dvc_pointer_path="data/processed/modeling_base.parquet.dvc",
        modeling_base_path="data/processed/modeling_base.parquet",
        records=2_000,
    )
    training = TrainingReference(
        modeling_base=modeling_base,
        random_forest_validation_macro_f1=0.75,
        selected_model_name="tfidf_random_forest",
    )
    optimization = OptimizationReference.from_xcom(
        {
            **modeling_base.to_xcom(),
            "final_speedup": 1.25,
            "onnx_gate_met": True,
            "selected_ccp_alpha": 0.0,
            "test_prediction_parity": 1.0,
        }
    )
    registration = RegistrationReference(
        model_name="triage-urgency-classifier",
        model_version="2",
        run_id="run-identifier",
    )

    payloads = [
        modeling_base.to_xcom(),
        training.to_xcom(),
        optimization.to_xcom(),
        registration.to_xcom(),
    ]

    assert ModelingBaseReference.from_xcom(modeling_base.to_xcom()) == modeling_base
    assert TrainingReference.from_xcom(training.to_xcom()) == training
    assert all(len(json.dumps(payload)) < 1_000 for payload in payloads)
    assert all("patient_text" not in json.dumps(payload) for payload in payloads)
    assert all("patient_id" not in json.dumps(payload) for payload in payloads)


def test_training_dag_has_the_required_linear_task_graph() -> None:
    dag_path = Path("dags/training_pipeline.py")
    spec = importlib.util.spec_from_file_location("kan14_training_dag", dag_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dag = module.training_dag

    assert dag.schedule is None
    assert set(dag.task_ids) == {
        "validate_modeling_base",
        "train_and_evaluate",
        "optimize_and_benchmark",
        "register_and_promote",
    }
    assert dag.get_task("train_and_evaluate").upstream_task_ids == {
        "validate_modeling_base"
    }
    assert dag.get_task("optimize_and_benchmark").upstream_task_ids == {
        "train_and_evaluate"
    }
    assert dag.get_task("register_and_promote").upstream_task_ids == {
        "optimize_and_benchmark"
    }


def test_manual_registration_creates_a_new_version_and_preserves_prior_version(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    dvc_pointer_path = tmp_path / "modeling_base.parquet.dvc"
    dvc_pointer_path.write_text(
        "outs:\n- md5: 0123456789abcdef0123456789abcdef\n"
        "  path: modeling_base.parquet\n",
        encoding="utf-8",
    )
    registry_config = RegistryConfig(
        model_name="manual-rerun-model",
        experiment_name="manual-rerun-experiment",
    )
    benchmark_config = OnnxBenchmarkConfig(
        benchmark_records=3,
        warmup_rounds=0,
        repetitions=1,
        desired_speedup=0.0001,
    )

    first = register_approved_onnx_model(
        _modeling_base(),
        baseline_config=_baseline_config(),
        benchmark_config=benchmark_config,
        dvc_pointer_path=dvc_pointer_path,
        registry_config=registry_config,
        tracking_uri=tracking_uri,
    )
    monkeypatch.setattr(
        model_registry,
        "run_and_log_onnx_benchmark",
        _unexpected_kan10_execution,
    )
    second = register_approved_onnx_model(
        _modeling_base(),
        baseline_config=_baseline_config(),
        benchmark_config=benchmark_config,
        dvc_pointer_path=dvc_pointer_path,
        registry_config=registry_config,
        tracking_uri=tracking_uri,
        prior_approval=OnnxBenchmarkApproval(
            final_speedup=1.0,
            onnx_gate_met=True,
            selected_ccp_alpha=0.0,
            test_prediction_parity=1.0,
        ),
    )

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    prior_version = client.get_model_version(first.model_name, first.model_version)
    champion = client.get_model_version_by_alias(first.model_name, "champion")

    assert first.run_id != second.run_id
    assert first.model_version != second.model_version
    assert prior_version.status == "READY"
    assert champion.version == second.model_version


def _modeling_base() -> pl.DataFrame:
    rows: list[dict[str, str]] = []
    for split, records_per_label in (("train", 4), ("validation", 2), ("test", 2)):
        for label, token in (("first", "red"), ("second", "blue"), ("third", "green")):
            for index in range(records_per_label):
                rows.append(
                    {
                        "patient_text_en": f"{token} marker {split} {index}",
                        "urgency": label,
                        "split": split,
                        "duplicate_cluster": f"{split}-{label}-{index}",
                    }
                )
    return pl.DataFrame(rows)


def _baseline_config() -> BaselineNlpConfig:
    return BaselineNlpConfig(
        min_document_frequency=1,
        random_forest_estimators=5,
        random_forest_n_jobs=1,
    )


def _unexpected_kan10_execution(*args: object, **kwargs: object) -> object:
    raise AssertionError("KAN-13 must not rerun KAN-10 when approval is supplied")
