"""Safe, typed orchestration handoffs for the KAN-14 training DAG."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from techchallenge.baseline_nlp import (
    BaselineNlpConfig,
    BaselineSelection,
    baseline_selection,
    load_modeling_base,
    read_dvc_provenance,
    run_and_log_experiment,
    split_modeling_base,
)
from techchallenge.model_registry import RegistryConfig, register_approved_onnx_model
from techchallenge.modeling_base import validate_modeling_base_path
from techchallenge.onnx_benchmark import (
    OnnxBenchmarkApproval,
    OnnxBenchmarkConfig,
    benchmark_approval,
    run_and_log_onnx_benchmark,
)

XComValue: TypeAlias = str | int | float | bool
XComPayload: TypeAlias = dict[str, XComValue]


@dataclass(frozen=True)
class ModelingBaseReference:
    """Validated input paths and aggregate provenance for a DAG task boundary."""

    dvc_md5: str
    dvc_pointer_path: str
    modeling_base_path: str
    records: int

    def to_xcom(self) -> XComPayload:
        """Return a JSON-safe XCom payload without records or text content."""
        return {
            "dvc_md5": self.dvc_md5,
            "dvc_pointer_path": self.dvc_pointer_path,
            "modeling_base_path": self.modeling_base_path,
            "records": self.records,
        }

    @classmethod
    def from_xcom(cls, payload: XComPayload) -> ModelingBaseReference:
        """Restore a validated task reference from its constrained XCom payload."""
        return cls(
            dvc_md5=_required_str(payload, "dvc_md5"),
            dvc_pointer_path=_required_str(payload, "dvc_pointer_path"),
            modeling_base_path=_required_str(payload, "modeling_base_path"),
            records=_required_int(payload, "records"),
        )


@dataclass(frozen=True)
class TrainingReference:
    """Aggregate KAN-11 result required by the KAN-10 task."""

    modeling_base: ModelingBaseReference
    random_forest_validation_macro_f1: float
    selected_model_name: str

    def to_xcom(self) -> XComPayload:
        """Return only paths and non-sensitive aggregate model selection metadata."""
        return {
            **self.modeling_base.to_xcom(),
            "random_forest_validation_macro_f1": (
                self.random_forest_validation_macro_f1
            ),
            "selected_model_name": self.selected_model_name,
        }

    @classmethod
    def from_xcom(cls, payload: XComPayload) -> TrainingReference:
        """Restore a KAN-11 reference from its constrained XCom payload."""
        return cls(
            modeling_base=ModelingBaseReference.from_xcom(payload),
            random_forest_validation_macro_f1=_required_float(
                payload, "random_forest_validation_macro_f1"
            ),
            selected_model_name=_required_str(payload, "selected_model_name"),
        )


@dataclass(frozen=True)
class OptimizationReference:
    """Aggregate KAN-10 approval data required by the KAN-13 task."""

    modeling_base: ModelingBaseReference
    approval: OnnxBenchmarkApproval

    def to_xcom(self) -> XComPayload:
        """Return only paths and non-sensitive gate measurements."""
        return {
            **self.modeling_base.to_xcom(),
            "final_speedup": self.approval.final_speedup,
            "onnx_gate_met": self.approval.onnx_gate_met,
            "selected_ccp_alpha": self.approval.selected_ccp_alpha,
            "test_prediction_parity": self.approval.test_prediction_parity,
        }

    @classmethod
    def from_xcom(cls, payload: XComPayload) -> OptimizationReference:
        """Restore KAN-10 approval from its constrained XCom payload."""
        return cls(
            modeling_base=ModelingBaseReference.from_xcom(payload),
            approval=OnnxBenchmarkApproval(
                final_speedup=_required_float(payload, "final_speedup"),
                onnx_gate_met=_required_bool(payload, "onnx_gate_met"),
                selected_ccp_alpha=_required_float(payload, "selected_ccp_alpha"),
                test_prediction_parity=_required_float(
                    payload, "test_prediction_parity"
                ),
            ),
        )


@dataclass(frozen=True)
class RegistrationReference:
    """New registry version identity returned by the terminal DAG task."""

    model_name: str
    model_version: str
    run_id: str

    def to_xcom(self) -> XComPayload:
        """Return only the new MLflow run and registry version identifiers."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "run_id": self.run_id,
        }


def validate_training_inputs(
    modeling_base_path: Path, dvc_pointer_path: Path
) -> ModelingBaseReference:
    """Validate the fixed modeling input before executing any training run."""
    if not dvc_pointer_path.is_file():
        raise ValueError("DVC pointer for modeling base does not exist")
    records = validate_modeling_base_path(modeling_base_path)
    modeling_base = load_modeling_base(modeling_base_path)
    split_modeling_base(modeling_base)
    provenance = read_dvc_provenance(dvc_pointer_path)
    return ModelingBaseReference(
        dvc_md5=provenance.md5,
        dvc_pointer_path=str(dvc_pointer_path),
        modeling_base_path=str(modeling_base_path),
        records=records,
    )


def train_and_evaluate(
    reference: ModelingBaseReference,
    *,
    baseline_config: BaselineNlpConfig,
    tracking_uri: str | None = None,
) -> TrainingReference:
    """Execute KAN-11 once and return its aggregate validation selection."""
    result = run_and_log_experiment(
        load_modeling_base(Path(reference.modeling_base_path)),
        config=baseline_config,
        dvc_pointer_path=Path(reference.dvc_pointer_path),
        tracking_uri=tracking_uri,
    )
    selection = baseline_selection(result)
    return TrainingReference(
        modeling_base=reference,
        random_forest_validation_macro_f1=(selection.random_forest_validation_macro_f1),
        selected_model_name=selection.selected_model_name,
    )


def optimize_and_benchmark(
    reference: TrainingReference,
    *,
    baseline_config: BaselineNlpConfig,
    benchmark_config: OnnxBenchmarkConfig,
    tracking_uri: str | None = None,
) -> OptimizationReference:
    """Execute KAN-10 once using KAN-11's aggregate validation selection."""
    result = run_and_log_onnx_benchmark(
        load_modeling_base(Path(reference.modeling_base.modeling_base_path)),
        baseline_config=baseline_config,
        benchmark_config=benchmark_config,
        dvc_pointer_path=Path(reference.modeling_base.dvc_pointer_path),
        tracking_uri=tracking_uri,
        prior_selection=reference_to_selection(reference),
    )
    return OptimizationReference(
        modeling_base=reference.modeling_base,
        approval=benchmark_approval(result),
    )


def register_and_promote(
    reference: OptimizationReference,
    *,
    baseline_config: BaselineNlpConfig,
    benchmark_config: OnnxBenchmarkConfig,
    registry_config: RegistryConfig = RegistryConfig(),
    tracking_uri: str | None = None,
) -> RegistrationReference:
    """Execute KAN-13 once after KAN-10 gates have completed successfully."""
    result = register_approved_onnx_model(
        load_modeling_base(Path(reference.modeling_base.modeling_base_path)),
        baseline_config=baseline_config,
        benchmark_config=benchmark_config,
        dvc_pointer_path=Path(reference.modeling_base.dvc_pointer_path),
        registry_config=registry_config,
        tracking_uri=tracking_uri,
        prior_approval=reference.approval,
    )
    return RegistrationReference(
        model_name=result.model_name,
        model_version=result.model_version,
        run_id=result.run_id,
    )


def reference_to_selection(reference: TrainingReference) -> BaselineSelection:
    """Convert DAG-safe training metadata to the KAN-10 handoff type."""
    return BaselineSelection(
        selected_model_name=reference.selected_model_name,
        random_forest_validation_macro_f1=(reference.random_forest_validation_macro_f1),
    )


def _required_str(payload: XComPayload, key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"XCom field {key} must be a string")
    return value


def _required_int(payload: XComPayload, key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"XCom field {key} must be an integer")
    return value


def _required_float(payload: XComPayload, key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"XCom field {key} must be numeric")
    return float(value)


def _required_bool(payload: XComPayload, key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"XCom field {key} must be a boolean")
    return value
