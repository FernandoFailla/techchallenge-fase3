"""Manual KAN-14 training DAG with safe XCom handoffs and MLflow tracking."""

import os
from datetime import datetime
from pathlib import Path

from airflow.sdk import dag, task

from techchallenge.baseline_nlp import BaselineNlpConfig
from techchallenge.model_registry import RegistryConfig
from techchallenge.onnx_benchmark import OnnxBenchmarkConfig
from techchallenge.training_pipeline import (
    ModelingBaseReference,
    OptimizationReference,
    TrainingReference,
    XComPayload,
    validate_training_inputs,
)
from techchallenge.training_pipeline import (
    optimize_and_benchmark as run_optimization,
)
from techchallenge.training_pipeline import (
    register_and_promote as run_registration,
)
from techchallenge.training_pipeline import (
    train_and_evaluate as run_training,
)

PROJECT_ROOT = Path(os.environ.get("TECHCHALLENGE_ROOT", ".")).resolve()
MODELING_BASE_PATH = PROJECT_ROOT / "data/processed/modeling_base.parquet"
DVC_POINTER_PATH = MODELING_BASE_PATH.with_suffix(".parquet.dvc")


@dag(
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["kan-14", "mlflow", "training"],
)
def training_pipeline() -> None:
    """Run each approved training phase once on demand."""

    @task
    def validate_modeling_base() -> XComPayload:
        return validate_training_inputs(MODELING_BASE_PATH, DVC_POINTER_PATH).to_xcom()

    @task
    def train_and_evaluate(modeling_base: XComPayload) -> XComPayload:
        return run_training(
            ModelingBaseReference.from_xcom(modeling_base),
            baseline_config=BaselineNlpConfig(),
        ).to_xcom()

    @task
    def optimize_and_benchmark(training: XComPayload) -> XComPayload:
        return run_optimization(
            TrainingReference.from_xcom(training),
            baseline_config=BaselineNlpConfig(),
            benchmark_config=OnnxBenchmarkConfig(),
        ).to_xcom()

    @task
    def register_and_promote(optimization: XComPayload) -> XComPayload:
        return run_registration(
            OptimizationReference.from_xcom(optimization),
            baseline_config=BaselineNlpConfig(),
            benchmark_config=OnnxBenchmarkConfig(),
            registry_config=RegistryConfig(),
        ).to_xcom()

    modeling_base = validate_modeling_base()
    training = train_and_evaluate(modeling_base)
    optimization = optimize_and_benchmark(training)
    register_and_promote(optimization)


training_dag = training_pipeline()
