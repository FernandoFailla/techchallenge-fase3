"""Register the approved KAN-10 ONNX bundle without exposing source text."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final

import joblib  # type: ignore[import-untyped]
import mlflow
import numpy as np
import onnxruntime as ort
import pandas as pd  # type: ignore[import-untyped]
import polars as pl
from mlflow.models import ModelSignature
from mlflow.pyfunc.model import PythonModel, PythonModelContext
from mlflow.types.schema import ColSpec, Schema
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from techchallenge.baseline_nlp import (
    BaselineNlpConfig,
    DataProvenance,
    SplitData,
    read_dvc_provenance,
    split_modeling_base,
    train_tfidf_random_forest,
)
from techchallenge.onnx_benchmark import (
    OnnxBenchmarkApproval,
    OnnxBenchmarkConfig,
    benchmark_approval,
    pipeline_components,
    run_and_log_onnx_benchmark,
    serialize_onnx_classifier,
)
from techchallenge.tracking import get_tracking_uri

DEFAULT_EXPERIMENT_NAME: Final = "kan-13-mlflow-registry"
MODEL_NAME: Final = "triage-urgency-classifier"
CHAMPION_ALIAS: Final = "champion"
_MODEL_ARTIFACT_PATH: Final = "model"
_BUNDLE_ARTIFACTS: Final = (
    "vectorizer",
    "onnx_model",
    "class_mapping",
    "bundle_manifest",
)
_BUNDLE_FILENAMES: Final = {
    "vectorizer": "vectorizer.joblib",
    "onnx_model": "classifier.onnx",
    "class_mapping": "class_mapping.json",
    "bundle_manifest": "bundle_manifest.json",
}


@dataclass(frozen=True)
class RegistryConfig:
    """Immutable settings for one local Model Registry promotion."""

    model_name: str = MODEL_NAME
    champion_alias: str = CHAMPION_ALIAS
    experiment_name: str = DEFAULT_EXPERIMENT_NAME


@dataclass(frozen=True)
class BundleHashes:
    """SHA-256 hashes recorded for every registered bundle component."""

    vectorizer_sha256: str
    onnx_model_sha256: str
    class_mapping_sha256: str
    manifest_sha256: str
    source_code_sha256: str
    run_sha256: str


@dataclass(frozen=True)
class RegistryResult:
    """The Registry version promoted only after all integrity checks succeed."""

    model_name: str
    model_version: str
    run_id: str
    champion_alias: str
    hashes: BundleHashes


class OnnxUrgencyPyfunc(PythonModel):
    """Serve a KAN-10 TF-IDF vectorizer and ONNX classifier as one PyFunc."""

    def load_context(self, context: PythonModelContext) -> None:
        vectorizer = joblib.load(context.artifacts["vectorizer"])
        if not isinstance(vectorizer, TfidfVectorizer):
            raise ValueError("Registered vectorizer is not a TfidfVectorizer")
        class_mapping = _read_class_mapping(Path(context.artifacts["class_mapping"]))
        session = ort.InferenceSession(
            context.artifacts["onnx_model"], providers=["CPUExecutionProvider"]
        )
        self._vectorizer = vectorizer
        self._class_mapping = class_mapping
        self._input_name = session.get_inputs()[0].name
        self._session = session

    def predict(
        self,
        context: PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, object] | None = None,
    ) -> pd.DataFrame:
        """Return one urgency class per input text without retaining the inputs."""
        del context, params
        if list(model_input.columns) != ["text"]:
            raise ValueError("Model input must contain exactly one text column")
        texts = tuple(model_input["text"])
        if any(not isinstance(text, str) for text in texts):
            raise ValueError("Model input text values must be strings")
        vectors = self._vectorizer.transform(texts).toarray().astype(np.float32)
        labels = tuple(
            str(label)
            for label in self._session.run(None, {self._input_name: vectors})[0]
        )
        if any(label not in self._class_mapping.values() for label in labels):
            raise ValueError(
                "ONNX output contains a class outside the registered mapping"
            )
        return pd.DataFrame({"urgency": labels})


def register_approved_onnx_model(
    modeling_base: pl.DataFrame,
    *,
    baseline_config: BaselineNlpConfig,
    benchmark_config: OnnxBenchmarkConfig,
    dvc_pointer_path: Path,
    registry_config: RegistryConfig = RegistryConfig(),
    tracking_uri: str | None = None,
    prior_approval: OnnxBenchmarkApproval | None = None,
) -> RegistryResult:
    """Reproduce KAN-10, register its bundle, then assign `champion` if verified."""
    resolved_tracking_uri = tracking_uri or get_tracking_uri()
    approval = prior_approval
    if approval is None:
        approval = benchmark_approval(
            run_and_log_onnx_benchmark(
                modeling_base,
                baseline_config=baseline_config,
                benchmark_config=benchmark_config,
                dvc_pointer_path=dvc_pointer_path,
                tracking_uri=resolved_tracking_uri,
            )
        )
    _require_approved_kan10_result(
        onnx_gate_met=approval.onnx_gate_met,
        prediction_parity=approval.test_prediction_parity,
    )

    selected_config = replace(
        baseline_config, random_forest_ccp_alpha=approval.selected_ccp_alpha
    )
    splits = split_modeling_base(modeling_base)
    final_model = train_tfidf_random_forest(
        _combine_train_and_validation(splits.train, splits.validation), selected_config
    )
    vectorizer, classifier = pipeline_components(final_model)
    provenance = read_dvc_provenance(dvc_pointer_path)

    mlflow.set_tracking_uri(resolved_tracking_uri)
    mlflow.set_registry_uri(resolved_tracking_uri)
    mlflow.set_experiment(registry_config.experiment_name)
    with TemporaryDirectory() as temporary_directory:
        bundle_directory = Path(temporary_directory)
        bundle_paths = _write_bundle(
            bundle_directory, vectorizer, classifier, final_model
        )
        hashes = _bundle_hashes(bundle_paths)
        manifest = _build_manifest(
            provenance=provenance,
            baseline_config=selected_config,
            benchmark_config=benchmark_config,
            hashes=hashes,
        )
        _write_json(bundle_paths["bundle_manifest"], manifest)
        hashes = replace(
            hashes, manifest_sha256=_sha256_file(bundle_paths["bundle_manifest"])
        )
        hashes = replace(
            hashes,
            run_sha256=_run_sha256(
                provenance=provenance,
                baseline_config=selected_config,
                benchmark_config=benchmark_config,
                hashes=hashes,
            ),
        )
        with mlflow.start_run(run_name="register-approved-onnx-bundle") as active_run:
            run_id = active_run.info.run_id
            mlflow.log_params(
                _registry_parameters(
                    provenance=provenance,
                    baseline_config=selected_config,
                    benchmark_config=benchmark_config,
                    hashes=hashes,
                )
            )
            mlflow.log_metrics(
                {
                    "kan10.test_prediction_parity": (approval.test_prediction_parity),
                    "kan10.onnx_speedup": approval.final_speedup,
                    "bundle.components": float(len(_BUNDLE_ARTIFACTS)),
                }
            )
            manifest_path = bundle_directory / "registry_manifest.json"
            _write_json(manifest_path, manifest)
            mlflow.log_artifact(str(manifest_path), artifact_path="provenance")
            mlflow.pyfunc.log_model(
                name=_MODEL_ARTIFACT_PATH,
                python_model=OnnxUrgencyPyfunc(),
                artifacts={
                    artifact_name: str(bundle_paths[artifact_name])
                    for artifact_name in _BUNDLE_ARTIFACTS
                },
                signature=_model_signature(),
                metadata={
                    "privacy": "no_clinical_text_ids_or_fitted_vocabulary_logged"
                },
            )

        model_version = mlflow.register_model(
            model_uri=f"runs:/{run_id}/{_MODEL_ARTIFACT_PATH}",
            name=registry_config.model_name,
            await_registration_for=60,
            tags={"integrity.bundle_manifest_sha256": hashes.manifest_sha256},
        )

    _verify_registered_bundle(
        model_name=registry_config.model_name,
        model_version=model_version.version,
        run_id=run_id,
        expected_hashes=hashes,
        tracking_uri=resolved_tracking_uri,
    )
    client = mlflow.MlflowClient(tracking_uri=resolved_tracking_uri)
    client.set_registered_model_alias(
        registry_config.model_name,
        registry_config.champion_alias,
        model_version.version,
    )
    champion = client.get_model_version_by_alias(
        registry_config.model_name, registry_config.champion_alias
    )
    if champion.version != model_version.version:
        raise RuntimeError(
            "Champion alias does not resolve to the verified model version"
        )
    return RegistryResult(
        model_name=registry_config.model_name,
        model_version=model_version.version,
        run_id=run_id,
        champion_alias=registry_config.champion_alias,
        hashes=hashes,
    )


def _combine_train_and_validation(first: SplitData, second: SplitData) -> SplitData:
    return SplitData(
        texts=first.texts + second.texts,
        targets=first.targets + second.targets,
    )


def _require_approved_kan10_result(
    *, onnx_gate_met: bool, prediction_parity: float
) -> None:
    if not onnx_gate_met:
        raise ValueError("KAN-10 ONNX speedup gate must pass before registration")
    if prediction_parity != 1.0:
        raise ValueError("KAN-10 test prediction parity must be exactly one")


def _write_bundle(
    directory: Path,
    vectorizer: TfidfVectorizer,
    classifier: RandomForestClassifier,
    pipeline: Pipeline,
) -> dict[str, Path]:
    paths = {
        "vectorizer": directory / "vectorizer.joblib",
        "onnx_model": directory / "classifier.onnx",
        "class_mapping": directory / "class_mapping.json",
        "bundle_manifest": directory / "bundle_manifest.json",
    }
    joblib.dump(vectorizer, paths["vectorizer"])
    paths["onnx_model"].write_bytes(serialize_onnx_classifier(pipeline))
    _write_json(
        paths["class_mapping"],
        {str(index): str(label) for index, label in enumerate(classifier.classes_)},
    )
    return paths


def _bundle_hashes(paths: dict[str, Path]) -> BundleHashes:
    return BundleHashes(
        vectorizer_sha256=_sha256_file(paths["vectorizer"]),
        onnx_model_sha256=_sha256_file(paths["onnx_model"]),
        class_mapping_sha256=_sha256_file(paths["class_mapping"]),
        manifest_sha256="",
        source_code_sha256=_sha256_file(Path(__file__)),
        run_sha256="",
    )


def _build_manifest(
    *,
    provenance: DataProvenance,
    baseline_config: BaselineNlpConfig,
    benchmark_config: OnnxBenchmarkConfig,
    hashes: BundleHashes,
) -> dict[str, object]:
    return {
        "bundle_schema_version": 1,
        "data_provenance": asdict(provenance),
        "training_configuration": {
            "baseline": asdict(baseline_config),
            "benchmark": asdict(benchmark_config),
        },
        "hashes": {
            "class_mapping_sha256": hashes.class_mapping_sha256,
            "onnx_model_sha256": hashes.onnx_model_sha256,
            "source_code_sha256": hashes.source_code_sha256,
            "vectorizer_sha256": hashes.vectorizer_sha256,
        },
        "privacy": {
            "contains_clinical_text": False,
            "contains_record_identifiers": False,
            "fitted_vocabulary_exposed_in_tracking_metadata": False,
            "contains_secrets": False,
        },
    }


def _registry_parameters(
    *,
    provenance: DataProvenance,
    baseline_config: BaselineNlpConfig,
    benchmark_config: OnnxBenchmarkConfig,
    hashes: BundleHashes,
) -> dict[str, str]:
    return {
        "data.dvc_md5": provenance.md5,
        "data.dvc_pointer": provenance.pointer_path,
        "provenance.data_dvc_md5": provenance.md5,
        "model.class_mapping_sha256": hashes.class_mapping_sha256,
        "model.onnx_sha256": hashes.onnx_model_sha256,
        "model.vectorizer_sha256": hashes.vectorizer_sha256,
        "bundle.manifest_sha256": hashes.manifest_sha256,
        "provenance.source_code_sha256": hashes.source_code_sha256,
        "run.integrity_sha256": hashes.run_sha256,
        "privacy.fitted_vocabulary_logged": "false",
        **{
            f"baseline_config.{key}": str(value)
            for key, value in asdict(baseline_config).items()
        },
        **{
            f"benchmark_config.{key}": str(value)
            for key, value in asdict(benchmark_config).items()
        },
    }


def _verify_registered_bundle(
    *,
    model_name: str,
    model_version: str,
    run_id: str,
    expected_hashes: BundleHashes,
    tracking_uri: str,
) -> None:
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    registered_version = client.get_model_version(model_name, model_version)
    if registered_version.run_id != run_id:
        raise RuntimeError("Registered model version is not linked to its expected run")
    if registered_version.status != "READY":
        raise RuntimeError("Registered model version is not ready for integrity checks")
    downloaded_model_directory = Path(
        mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{model_name}/{model_version}",
            tracking_uri=tracking_uri,
            registry_uri=tracking_uri,
        )
    )
    downloaded_hashes = {
        artifact_name: _sha256_file(
            downloaded_model_directory / "artifacts" / _BUNDLE_FILENAMES[artifact_name]
        )
        for artifact_name in _BUNDLE_ARTIFACTS
    }
    if downloaded_hashes != {
        "vectorizer": expected_hashes.vectorizer_sha256,
        "onnx_model": expected_hashes.onnx_model_sha256,
        "class_mapping": expected_hashes.class_mapping_sha256,
        "bundle_manifest": expected_hashes.manifest_sha256,
    }:
        raise RuntimeError(
            "Registered model bundle hashes do not match the run manifest"
        )
    mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")


def _model_signature() -> ModelSignature:
    return ModelSignature(
        inputs=Schema([ColSpec("string", "text")]),
        outputs=Schema([ColSpec("string", "urgency")]),
    )


def _read_class_mapping(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Registered class mapping must be a non-empty object")
    mapping = {str(key): str(value) for key, value in payload.items()}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Registered class mapping must have unique labels")
    return mapping


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_sha256(
    *,
    provenance: DataProvenance,
    baseline_config: BaselineNlpConfig,
    benchmark_config: OnnxBenchmarkConfig,
    hashes: BundleHashes,
) -> str:
    payload = {
        "baseline_config": asdict(baseline_config),
        "benchmark_config": asdict(benchmark_config),
        "data_dvc_md5": provenance.md5,
        "hashes": {
            "class_mapping_sha256": hashes.class_mapping_sha256,
            "manifest_sha256": hashes.manifest_sha256,
            "onnx_model_sha256": hashes.onnx_model_sha256,
            "source_code_sha256": hashes.source_code_sha256,
            "vectorizer_sha256": hashes.vectorizer_sha256,
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
