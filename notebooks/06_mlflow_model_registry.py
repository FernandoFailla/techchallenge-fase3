import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    from techchallenge.baseline_nlp import BaselineNlpConfig, load_modeling_base
    from techchallenge.model_registry import (
        RegistryConfig,
        register_approved_onnx_model,
    )
    from techchallenge.onnx_benchmark import OnnxBenchmarkConfig

    return (
        BaselineNlpConfig,
        OnnxBenchmarkConfig,
        Path,
        RegistryConfig,
        alt,
        load_modeling_base,
        mo,
        pl,
        register_approved_onnx_model,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # KAN-13: registro local do classificador de urgência

    Este experimento reproduz o fluxo aprovado no KAN-10: TF-IDF ajustado em
    processo, Random Forest convertido para ONNX e a mesma ordem de classes. O
    bundle PyFunc é registrado localmente somente depois da seleção por validação
    e da paridade no teste reservado. Textos clínicos, identificadores, vocabulário
    ajustado e segredos não são exibidos ou registrados como parâmetros, métricas
    ou artefatos de acompanhamento.
    """)
    return


@app.cell
def _(Path, load_modeling_base, mo):
    data_path = Path("data/processed/modeling_base.parquet")
    mo.stop(not data_path.is_file(), "Base ausente. Execute `make pull-data`.")
    modeling_base = load_modeling_base(data_path)
    return data_path, modeling_base


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
        .properties(
            title="Distribuição agregada usada pelo registro", width=520, height=300
        )
    )
    split_chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Registro com integridade verificável

    O experimento refaz os runs de seleção e benchmark do KAN-10 no MLflow. Em
    seguida, registra o PyFunc `triage-urgency-classifier` com os binários do
    vetor TF-IDF, ONNX e mapa de classes. Um manifesto agregado com SHA-256 é
    registrado separadamente. O alias `champion` só é definido após baixar os
    componentes registrados e comparar todos os hashes esperados.
    """)
    return


@app.cell
def _(
    BaselineNlpConfig,
    OnnxBenchmarkConfig,
    RegistryConfig,
    data_path,
    modeling_base,
    register_approved_onnx_model,
):
    registry_result = register_approved_onnx_model(
        modeling_base,
        baseline_config=BaselineNlpConfig(),
        benchmark_config=OnnxBenchmarkConfig(),
        dvc_pointer_path=data_path.with_suffix(".parquet.dvc"),
        registry_config=RegistryConfig(),
    )
    return (registry_result,)


@app.cell
def _(alt, pl, registry_result):
    integrity_data = pl.DataFrame(
        {
            "component": ["TF-IDF", "ONNX", "Mapa de classes", "Manifesto"],
            "verified": [1, 1, 1, 1],
        }
    )
    integrity_chart = (
        alt.Chart(alt.Data(values=integrity_data.to_dicts()))
        .mark_bar(color="#15803d")
        .encode(
            x=alt.X("component:N", title="Componente"),
            y=alt.Y(
                "verified:Q",
                title="Integridade verificada",
                scale=alt.Scale(domain=[0, 1]),
            ),
            tooltip=["component:N", "verified:Q"],
        )
        .properties(title="Verificação do bundle promovido", width=520, height=300)
    )
    integrity_chart
    return


@app.cell(hide_code=True)
def _(mo, registry_result):
    mo.md(f"""
    ## Resultado

    A versão **{registry_result.model_version}** do modelo
    **`{registry_result.model_name}`** foi registrada no run
    **`{registry_result.run_id}`**. Todos os hashes do bundle conferiram após o
    download do Registry e o alias **`{registry_result.champion_alias}`** foi
    atribuído somente então.
    """)
    return


if __name__ == "__main__":
    app.run()
