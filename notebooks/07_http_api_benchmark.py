import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo

    from techchallenge.http_benchmark import (
        HttpBenchmarkConfig,
        run_and_log_http_benchmark,
    )

    return HttpBenchmarkConfig, alt, mo, run_and_log_http_benchmark


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # KAN-24: benchmark HTTP sequencial da API

    Este experimento mede a latência observada pelo cliente para chamadas `POST /predict`
    contra o contêiner local em `http://localhost:8000`. Ele faz 20 aquecimentos e 200
    requisições sequenciais com uma entrada sintética determinística. O benchmark não
    persiste texto de requisição, identificadores nem payloads de resposta; o MLflow recebe
    somente versão do modelo, configuração e métricas agregadas.
    """)
    return


@app.cell
def _(HttpBenchmarkConfig, run_and_log_http_benchmark):
    benchmark_result = run_and_log_http_benchmark(config=HttpBenchmarkConfig())
    return (benchmark_result,)


@app.cell
def _(alt, benchmark_result):
    latency_rows = [
        {"metric": "Média", "latency_ms": benchmark_result.mean_latency_ms},
        {"metric": "P50", "latency_ms": benchmark_result.p50_latency_ms},
        {"metric": "P95", "latency_ms": benchmark_result.p95_latency_ms},
    ]
    latency_chart = (
        alt.Chart(alt.Data(values=latency_rows))
        .mark_bar(color="#0f766e")
        .encode(
            x=alt.X("metric:N", title="Métrica de latência"),
            y=alt.Y("latency_ms:Q", title="Latência HTTP (ms)"),
            tooltip=["metric:N", alt.Tooltip("latency_ms:Q", format=".3f")],
        )
        .properties(
            title="Latência HTTP agregada no contêiner local", width=420, height=300
        )
    )
    latency_chart
    return


@app.cell(hide_code=True)
def _(benchmark_result, mo):
    mo.md(f"""
    ## Resultado

    A versão **{benchmark_result.model_version}** respondeu às
    **{benchmark_result.measured_requests}** requisições sequenciais após
    **{benchmark_result.warmup_requests}** aquecimentos. A média foi
    **{benchmark_result.mean_latency_ms:.3f} ms**, com P95 de
    **{benchmark_result.p95_latency_ms:.3f} ms**. Estes números dependem da máquina
    local, da imagem em execução e da versão do modelo indicada acima.
    """)
    return


if __name__ == "__main__":
    app.run()
