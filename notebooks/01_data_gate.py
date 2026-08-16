import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import polars as pl

    return Path, mo, pl


@app.cell
def _(Path, pl):
    dataset_path = Path("data/raw/synthetic_v1.csv")
    dataset = pl.read_csv(dataset_path)
    dataset
    return dataset, dataset_path


@app.cell
def _(dataset):
    dataset_drop = dataset.drop(["patient_text_ckb", "patient_text_fa"])
    dataset_drop
    return (dataset_drop,)


@app.cell
def _(dataset_drop):
    dataset_drop.describe()
    return


@app.cell(hide_code=True)
def conclusions(mo):
    mo.md("""
    ## Conclusão da inspeção

    - O arquivo `synthetic_v1.csv` foi carregado com **2.000 registros** e 12 colunas.
    - A coluna `patient_text_en` está disponível para ser usada como entrada do
      classificador de NLP.
    - A coluna `urgency` está disponível como target publicado, com as classes
      `low`, `medium` e `high`.
    - Não foram encontrados valores nulos nas duas colunas necessárias para a
      classificação.
    - As colunas de texto em curdo e persa foram removidas da visualização de
      trabalho, mantendo o texto em inglês para o escopo do MVP.

    Portanto, a estrutura do dataset atende à necessidade inicial de uma
    classificação textual de urgência em três classes. As verificações de
    licença, rastreabilidade, duplicatas e separação sem vazamento serão
    registradas nas próximas etapas do gate de dados.
    """)
    return


if __name__ == "__main__":
    app.run()
