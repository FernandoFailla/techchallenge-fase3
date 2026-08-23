import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    from techchallenge.baseline_nlp import (
        BaselineNlpConfig,
        load_modeling_base,
        run_and_log_experiment,
    )

    return (
        BaselineNlpConfig,
        Path,
        alt,
        load_modeling_base,
        mo,
        pl,
        run_and_log_experiment,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # KAN-11: baseline NLP reproduzível

    Este experimento compara um baseline de classe majoritária com TF-IDF e
    Random Forest. Ele lê somente a base DVC `data/processed/modeling_base.parquet`.
    Os textos permanecem em memória durante o ajuste: a interface, os logs e os
    artefatos do MLflow exibem somente métricas e contagens agregadas.
    """)
    return


@app.cell
def _(Path, load_modeling_base, mo):
    data_path = Path("data/processed/modeling_base.parquet")
    mo.stop(not data_path.is_file(), "Base ausente. Execute `make pull-data`.")
    modeling_base = load_modeling_base(data_path)

    return data_path, modeling_base


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Divisão reservada

    O modelo é ajustado em `train`, comparado em `validation` e apenas o vencedor é
    reajustado com `train + validation` para a medição final em `test`. O teste não
    orienta escolha de modelo ou hiperparâmetros.
    """)
    return


@app.cell
def _(alt, modeling_base, pl):
    split_counts = (
        modeling_base.group_by(["split", "urgency"])
        .len(name="records")
        .sort(["split", "urgency"])
    )
    split_chart = (
        alt.Chart(alt.Data(values=split_counts.to_dicts()))
        .mark_bar()
        .encode(
            x=alt.X("split:N", sort=["train", "validation", "test"], title="Conjunto"),
            xOffset=alt.XOffset("urgency:N", title="Urgência"),
            y=alt.Y("records:Q", title="Registros"),
            color=alt.Color("urgency:N", title="Urgência"),
            tooltip=["split:N", "urgency:N", "records:Q"],
        )
        .properties(title="Distribuição agregada por conjunto", width=520, height=300)
    )

    split_chart

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Ajuste e rastreamento

    Cada execução registra parâmetros, métricas agregadas, matriz de confusão e a
    proveniência do ponteiro DVC no MLflow. O modelo e a matriz TF-IDF não são
    registrados porque podem reter tokens do texto de entrada.
    """)
    return


@app.cell
def _(BaselineNlpConfig, data_path, modeling_base, run_and_log_experiment):
    experiment_result = run_and_log_experiment(
        modeling_base,
        config=BaselineNlpConfig(),
        dvc_pointer_path=data_path.with_suffix(".parquet.dvc"),
    )

    return (experiment_result,)


@app.cell
def _(alt, experiment_result, pl):
    validation_scores = pl.DataFrame(
        {
            "model": [
                experiment_result.validation_results[0].model_name,
                experiment_result.validation_results[1].model_name,
            ],
            "macro_f1": [
                experiment_result.validation_results[0].metrics["macro_f1"],
                experiment_result.validation_results[1].metrics["macro_f1"],
            ],
        }
    )
    validation_chart = (
        alt.Chart(alt.Data(values=validation_scores.to_dicts()))
        .mark_bar(color="#2563eb")
        .encode(
            x=alt.X("model:N", title="Modelo"),
            y=alt.Y("macro_f1:Q", title="Macro-F1", scale=alt.Scale(domain=[0, 1])),
            tooltip=["model:N", alt.Tooltip("macro_f1:Q", format=".3f")],
        )
        .properties(title="Seleção por Macro-F1 na validação", width=420, height=300)
    )

    validation_chart

    return


@app.cell(hide_code=True)
def _(experiment_result, mo):
    final_metrics = experiment_result.final_test_result.metrics
    mo.md(f"""
    ## Resultado final

    O modelo selecionado foi **{experiment_result.selected_model_name}**. No teste
    reservado, ele obteve **Macro-F1 de {final_metrics['macro_f1']:.3f}** e
    **acurácia de {final_metrics['accuracy']:.3f}**. A matriz de confusão agregada
    está disponível somente como artefato seguro no MLflow.
    """)
    return


if __name__ == "__main__":
    app.run()
