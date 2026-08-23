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
    from techchallenge.onnx_benchmark import (
        OnnxBenchmarkConfig,
        run_and_log_onnx_benchmark,
    )

    return (
        BaselineNlpConfig,
        OnnxBenchmarkConfig,
        Path,
        alt,
        load_modeling_base,
        mo,
        pl,
        run_and_log_onnx_benchmark,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # KAN-10: benchmark ONNX do classificador de urgência

    O experimento reutiliza o Random Forest TF-IDF do KAN-11 e a base DVC
    `data/processed/modeling_base.parquet`. Apenas o classificador é convertido
    para ONNX; a vetorização TF-IDF continua no processo. Textos ficam somente em
    memória, inclusive durante o benchmark, e o MLflow recebe métricas,
    parâmetros, proveniência e artefatos agregados.
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
    ## Critério de seleção

    O KAN-11 seleciona o candidato por Macro-F1 na validação. O benchmark usa
    referências determinísticas dessa mesma validação. Se a versão ONNX não
    alcançar o speedup desejado, o experimento avalia `ccp_alpha` somente na
    validação, preservando Macro-F1. O teste reservado não escolhe modelo nem
    hiperparâmetro: ele verifica apenas a paridade de classes do modelo final.
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
        .properties(title="Distribuição agregada da base DVC", width=520, height=300)
    )
    split_chart
    return


@app.cell
def _(
    BaselineNlpConfig,
    OnnxBenchmarkConfig,
    data_path,
    modeling_base,
    run_and_log_onnx_benchmark,
):
    benchmark_result = run_and_log_onnx_benchmark(
        modeling_base,
        baseline_config=BaselineNlpConfig(),
        benchmark_config=OnnxBenchmarkConfig(),
        dvc_pointer_path=data_path.with_suffix(".parquet.dvc"),
    )
    return (benchmark_result,)


@app.cell
def _(alt, benchmark_result, pl):
    latency_data = pl.DataFrame(
        {
            "implementation": ["scikit-learn", "ONNX Runtime"],
            "mean_latency_ms": [
                benchmark_result.final_benchmark.sklearn_mean_latency_ms,
                benchmark_result.final_benchmark.onnx_mean_latency_ms,
            ],
        }
    )
    latency_chart = (
        alt.Chart(alt.Data(values=latency_data.to_dicts()))
        .mark_bar(color="#0f766e")
        .encode(
            x=alt.X("implementation:N", title="Fluxo completo"),
            y=alt.Y("mean_latency_ms:Q", title="Latência média por texto (ms)"),
            tooltip=["implementation:N", alt.Tooltip("mean_latency_ms:Q", format=".4f")],
        )
        .properties(title="Benchmark em processo: texto para classe", width=420, height=300)
    )
    latency_chart
    return


@app.cell(hide_code=True)
def _(benchmark_result, mo):
    final_benchmark = benchmark_result.final_benchmark
    gate_status = "atingiu" if benchmark_result.onnx_gate_met else "não atingiu"
    pruning_status = "foi avaliado" if benchmark_result.pruning_attempted else "não foi necessário"
    mo.md(f"""
    ## Resultado

    O ONNX Runtime {gate_status} o gate de speedup de
    **{final_benchmark.speedup:.3f}x** no conjunto de referências de validação.
    O fallback de pruning {pruning_status}; o `ccp_alpha` final é
    **{benchmark_result.selected_ccp_alpha:.6f}**. A paridade no teste reservado
    foi **{benchmark_result.test_prediction_parity.parity_rate:.3f}** em
    **{benchmark_result.test_prediction_parity.records}** registros.
    """)
    return


if __name__ == "__main__":
    app.run()
