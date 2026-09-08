import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import mlflow
    import numpy as np
    import pandas as pd
    import polars as pl
    import shap
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer

    from techchallenge.baseline_nlp import (
        BaselineNlpConfig,
        SplitData,
        evaluate_classifier,
        evaluate_predictions,
        load_modeling_base,
        split_modeling_base,
        train_dummy_classifier,
        train_tfidf_random_forest,
    )
    from techchallenge.model_registry import CHAMPION_ALIAS, MODEL_NAME
    from techchallenge.tracking import get_tracking_uri

    return (
        BaselineNlpConfig,
        CHAMPION_ALIAS,
        MODEL_NAME,
        Path,
        RandomForestClassifier,
        SplitData,
        TfidfVectorizer,
        alt,
        evaluate_classifier,
        evaluate_predictions,
        get_tracking_uri,
        json,
        load_modeling_base,
        mlflow,
        mo,
        np,
        pd,
        pl,
        shap,
        split_modeling_base,
        train_dummy_classifier,
        train_tfidf_random_forest,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Avaliacao e interpretabilidade do champion

    Este notebook compara o `champion` carregado do MLflow com o baseline de
    classe majoritaria no conjunto reservado. Ele tambem reconstrói o Random
    Forest de referencia com a configuracao registrada no manifesto do champion
    para exibir termos TF-IDF, n-grams e atribuicoes SHAP globais.

    **Autorizacao e limite:** por solicitacao explicita, termos e n-grams podem
    ser mostrados nesta analise local. Nao sao enviados para MLflow, logs ou
    arquivos de saida. O notebook nunca mostra textos completos, IDs ou exemplos
    de registros.
    """)
    return


@app.cell
def _(Path, load_modeling_base, mo):
    data_path = Path("data/processed/modeling_base.parquet")
    mo.stop(not data_path.is_file(), "Base ausente. Execute `make pull-data`.")
    modeling_base = load_modeling_base(data_path)
    return data_path, modeling_base


@app.cell
def _(
    CHAMPION_ALIAS,
    MODEL_NAME,
    Path,
    get_tracking_uri,
    json,
    mlflow,
    mo,
):
    tracking_uri = get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    registry_client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    champion_version = registry_client.get_model_version_by_alias(
        MODEL_NAME, CHAMPION_ALIAS
    ).version
    champion_uri = f"models:/{MODEL_NAME}/{champion_version}"
    champion_model = mlflow.pyfunc.load_model(champion_uri)
    downloaded_model_path = Path(
        mlflow.artifacts.download_artifacts(
            artifact_uri=champion_uri,
            tracking_uri=tracking_uri,
            registry_uri=tracking_uri,
        )
    )
    manifest_path = downloaded_model_path / "artifacts" / "bundle_manifest.json"
    mo.stop(not manifest_path.is_file(), "Manifesto do champion ausente.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return champion_model, champion_version, manifest


@app.cell
def _(
    BaselineNlpConfig,
    SplitData,
    champion_model,
    evaluate_classifier,
    evaluate_predictions,
    manifest,
    modeling_base,
    mo,
    pd,
    split_modeling_base,
    train_dummy_classifier,
    train_tfidf_random_forest,
):
    training_configuration = manifest.get("training_configuration")
    mo.stop(
        not isinstance(training_configuration, dict),
        "Manifesto do champion sem configuracao de treinamento.",
    )
    baseline_configuration = training_configuration.get("baseline")
    mo.stop(
        not isinstance(baseline_configuration, dict),
        "Manifesto do champion sem configuracao do baseline.",
    )
    baseline_config = BaselineNlpConfig(**baseline_configuration)
    splits = split_modeling_base(modeling_base)
    train_validation = SplitData(
        texts=splits.train.texts + splits.validation.texts,
        targets=splits.train.targets + splits.validation.targets,
    )
    baseline_model = train_dummy_classifier(train_validation)
    reference_model = train_tfidf_random_forest(train_validation, baseline_config)
    baseline_result = evaluate_classifier(
        baseline_model,
        splits.test,
        labels=splits.labels,
        model_name="dummy_majority",
        split_name="test",
    )
    reference_result = evaluate_classifier(
        reference_model,
        splits.test,
        labels=splits.labels,
        model_name="random_forest_reference",
        split_name="test",
    )
    champion_payload = champion_model.predict(pd.DataFrame({"text": splits.test.texts}))
    mo.stop(
        list(champion_payload.columns) != ["urgency"],
        "Champion retornou um contrato de previsao invalido.",
    )
    champion_predictions = tuple(str(value) for value in champion_payload["urgency"])
    champion_result = evaluate_predictions(
        splits.test.targets,
        champion_predictions,
        labels=splits.labels,
        model_name="champion_onnx",
        split_name="test",
    )
    reference_predictions = tuple(
        str(value) for value in reference_model.predict(splits.test.texts)
    )
    reference_agreement = sum(
        champion == reference
        for champion, reference in zip(champion_predictions, reference_predictions)
    ) / len(champion_predictions)
    mo.stop(
        reference_agreement != 1.0,
        "A referencia Random Forest nao corresponde integralmente ao champion ONNX; "
        "interpretabilidade bloqueada para evitar uma explicacao enganosa.",
    )
    return (
        baseline_model,
        baseline_result,
        champion_predictions,
        champion_result,
        reference_agreement,
        reference_model,
        reference_result,
        splits,
    )


@app.cell
def _(alt, baseline_result, champion_result, pl, reference_result):
    comparison_data = pl.DataFrame(
        {
            "model": ["Baseline", "Referencia RF", "Champion ONNX"],
            "macro_f1": [
                baseline_result.metrics["macro_f1"],
                reference_result.metrics["macro_f1"],
                champion_result.metrics["macro_f1"],
            ],
        }
    )
    comparison_chart = (
        alt.Chart(alt.Data(values=comparison_data.to_dicts()))
        .mark_bar()
        .encode(
            x=alt.X("model:N", title="Modelo"),
            y=alt.Y("macro_f1:Q", title="Macro-F1 no teste", scale=alt.Scale(zero=True)),
            color=alt.Color("model:N", legend=None),
            tooltip=["model:N", alt.Tooltip("macro_f1:Q", format=".3f")],
        )
        .properties(title="Champion comparado ao baseline", width=520, height=300)
    )
    comparison_chart
    return


@app.cell
def _(alt, champion_predictions, pl, splits):
    prediction_distribution = (
        pl.DataFrame({"classification": champion_predictions})
        .lazy()
        .group_by("classification")
        .len(name="records")
        .rename({"classification": "urgency"})
        .join(
            pl.DataFrame({"urgency": splits.labels}).lazy(),
            on="urgency",
            how="right",
        )
        .with_columns(pl.col("records").fill_null(0))
        .sort("urgency")
        .collect()
    )
    distribution_chart = (
        alt.Chart(alt.Data(values=prediction_distribution.to_dicts()))
        .mark_bar(color="#2563eb")
        .encode(
            x=alt.X("urgency:N", title="Classe prevista"),
            y=alt.Y("records:Q", title="Predicoes no teste", scale=alt.Scale(zero=True)),
            tooltip=["urgency:N", "records:Q"],
        )
        .properties(title="Distribuicao de previsoes do champion", width=420, height=300)
    )
    distribution_chart
    return


@app.cell
def _(alt, champion_result, pl):
    confusion_rows = [
        {
            "actual": actual,
            "predicted": predicted,
            "records": champion_result.confusion_matrix[row_index][column_index],
        }
        for row_index, actual in enumerate(champion_result.labels)
        for column_index, predicted in enumerate(champion_result.labels)
    ]
    confusion_chart = (
        alt.Chart(alt.Data(values=pl.DataFrame(confusion_rows).to_dicts()))
        .mark_rect()
        .encode(
            x=alt.X("predicted:N", title="Classe prevista"),
            y=alt.Y("actual:N", title="Classe real"),
            color=alt.Color("records:Q", title="Registros"),
            tooltip=["actual:N", "predicted:N", "records:Q"],
        )
        .properties(title="Matriz de confusao do champion", width=420, height=320)
    )
    confusion_chart
    return


@app.cell
def _(RandomForestClassifier, TfidfVectorizer, alt, pl, reference_model):
    vectorizer = reference_model.named_steps["tfidf"]
    random_forest = reference_model.named_steps["random_forest"]
    if not isinstance(vectorizer, TfidfVectorizer) or not isinstance(
        random_forest, RandomForestClassifier
    ):
        raise TypeError("Pipeline de referencia nao contem TF-IDF e Random Forest")
    tfidf_importance = (
        pl.DataFrame(
            {
                "term": vectorizer.get_feature_names_out().tolist(),
                "importance": random_forest.feature_importances_.tolist(),
            }
        )
        .lazy()
        .sort("importance", descending=True)
        .head(20)
        .sort("importance")
        .collect()
    )
    tfidf_chart = (
        alt.Chart(alt.Data(values=tfidf_importance.to_dicts()))
        .mark_bar(color="#7c3aed")
        .encode(
            x=alt.X("importance:Q", title="Importancia do Random Forest"),
            y=alt.Y("term:N", sort=None, title="Termo ou n-gram"),
            tooltip=["term:N", alt.Tooltip("importance:Q", format=".5f")],
        )
        .properties(title="20 termos TF-IDF mais importantes", width=520, height=520)
    )
    tfidf_chart
    return random_forest, vectorizer


@app.cell
def _(alt, np, pl, random_forest, shap, splits, vectorizer):
    explanation_records = min(64, splits.test.records)
    explanation_matrix = vectorizer.transform(
        splits.test.texts[:explanation_records]
    ).toarray()
    explainer = shap.TreeExplainer(random_forest, feature_perturbation="tree_path_dependent")
    shap_values = np.asarray(
        explainer.shap_values(explanation_matrix, check_additivity=False)
    )
    if shap_values.ndim != 3:
        raise ValueError("SHAP retornou uma forma inesperada para classificacao multiclasse")
    shap_importance = (
        pl.DataFrame(
            {
                "term": vectorizer.get_feature_names_out().tolist(),
                "mean_absolute_shap": np.abs(shap_values).mean(axis=(0, 2)).tolist(),
            }
        )
        .lazy()
        .sort("mean_absolute_shap", descending=True)
        .head(20)
        .sort("mean_absolute_shap")
        .collect()
    )
    shap_chart = (
        alt.Chart(alt.Data(values=shap_importance.to_dicts()))
        .mark_bar(color="#db2777")
        .encode(
            x=alt.X("mean_absolute_shap:Q", title="Media absoluta SHAP"),
            y=alt.Y("term:N", sort=None, title="Termo ou n-gram"),
            tooltip=["term:N", alt.Tooltip("mean_absolute_shap:Q", format=".5f")],
        )
        .properties(title="20 termos com maior atribuicao SHAP global", width=520, height=520)
    )
    shap_chart
    return


@app.cell(hide_code=True)
def _(champion_result, champion_version, mo, reference_agreement):
    mo.md(f"""
    ## Conclusao

    O champion ONNX na versao **{champion_version}** atingiu Macro-F1 de
    **{champion_result.metrics["macro_f1"]:.3f}** no teste reservado. A referencia
    Random Forest teve concordancia de **{reference_agreement:.3f}** com o champion
    antes da geracao das explicacoes. Os graficos TF-IDF e SHAP descrevem, portanto,
    essa referencia equivalente; eles nao sao explicacoes nativas do runtime ONNX.

    Termos e n-grams exibidos aqui sao autorizados somente para esta analise local.
    Nao os exporte para MLflow, logs, Grafana, relatorios publicos ou ambientes que
    recebam dados reais.
    """)
    return


if __name__ == "__main__":
    app.run()
