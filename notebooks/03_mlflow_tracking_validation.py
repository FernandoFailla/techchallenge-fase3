import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import mlflow

    from techchallenge.tracking import get_tracking_uri

    return get_tracking_uri, mlflow, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Validação do MLflow Tracking

    Esta verificação registra apenas metadados sintéticos para confirmar a conexão
    local. Nenhum texto clínico, modelo ou dado do dataset é enviado ao MLflow.
    """)
    return


@app.cell
def _(get_tracking_uri, mlflow):
    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.set_experiment("tracking-validation")
    with mlflow.start_run(run_name="connectivity-check"):
        mlflow.log_param("purpose", "connectivity-check")
        mlflow.log_metric("synthetic_metric", 1.0)

    return


if __name__ == "__main__":
    app.run()
