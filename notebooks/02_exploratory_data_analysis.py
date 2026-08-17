import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import NearestNeighbors


    return (
        NearestNeighbors,
        Path,
        TfidfVectorizer,
        alt,
        mo,
        pl,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # EDA do KurMed-Triage

    Queremos saber se esta base permite treinar, com segurança metodológica, um
    classificador ordinal de urgência usando a queixa em inglês.

    O raciocínio seguirá uma pergunta por vez. Cada pergunta é apresentada antes do
    gráfico; a interpretação aparece logo abaixo. Nenhum texto clínico individual é
    exibido.
    """)
    return


@app.cell
def _(Path, mo, pl):
    dataset_path = Path("data/raw/synthetic_v1.csv")
    mo.stop(not dataset_path.is_file(), "Dataset ausente. Execute `make pull-data`.")
    dataset = pl.read_csv(dataset_path)

    return (dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. A base contém o necessário para definir a tarefa e separar os dados?

    Antes de olhar distribuições, verificamos quais campos existem. Texto e target
    são suficientes para um baseline, mas paciente e tempo são necessários para
    evitar que informações relacionadas atravessem treino e teste.
    """)
    return


@app.cell
def _(alt, pl):
    field_readiness = pl.DataFrame(
        {
            "field": ["Text", "Target", "Record ID", "Patient ID", "Timestamp", "Institution"],
            "status": ["Available", "Available", "Available", "Missing", "Missing", "Missing"],
            "order": [1, 2, 3, 4, 5, 6],
        }
    )

    field_chart = (
        alt.Chart(field_readiness)
        .mark_circle(size=420)
        .encode(
            y=alt.Y("field:N", sort=alt.SortField("order"), title=None),
            x=alt.X("status:N", title=None),
            color=alt.Color(
                "status:N",
                scale=alt.Scale(domain=["Available", "Missing"], range=["#22c55e", "#ef4444"]),
                legend=None,
            ),
            tooltip=["field:N", "status:N"],
        )
        .properties(title="Campos disponíveis para o desenho do experimento", width=420, height=240)
        .interactive(name="field-readiness-zoom")
    )

    field_chart

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### O que isso significa

    A base permite um baseline com `patient_text_en` como entrada e `urgency` como
    target. Porém, não há `patient_id`, `timestamp` ou instituição.

    Com esses metadados seria possível agrupar documentos do mesmo paciente, fazer
    uma divisão temporal e testar generalização entre instituições. Sem eles, essas
    garantias não podem ser demonstradas. Por isso, metadados estruturados não serão
    features do MVP e essa limitação deverá acompanhar os resultados.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. As três classes estão muito desbalanceadas?

    Se uma classe dominar a base, acurácia pode parecer boa mesmo quando o modelo
    ignora urgências menos frequentes. Primeiro comparamos o volume das três classes.
    """)
    return


@app.cell
def _(alt, dataset):
    label_distribution = dataset.group_by("urgency").len(name="records")

    label_chart = (
        alt.Chart(alt.Data(values=label_distribution.to_dicts()))
        .mark_bar(size=38, opacity=1)
        .encode(
            y=alt.Y("urgency:N", sort=["low", "medium", "high"], title="Urgência"),
            x=alt.X(
                "records:Q",
                title="Registros",
                scale=alt.Scale(domain=[0, 850]),
            ),
            color=alt.Color(
                "urgency:N",
                scale=alt.Scale(
                    domain=["low", "medium", "high"],
                    range=["#22c55e", "#f59e0b", "#ef4444"],
                ),
                legend=None,
            ),
            tooltip=[alt.Tooltip("urgency:N"), alt.Tooltip("records:Q")],
        )
        .properties(title="Registros por classe", width=480, height=220)
        .interactive(name="label-distribution-zoom", bind_x=True, bind_y=False)
    )

    label_chart

    return (label_distribution,)


@app.cell(hide_code=True)
def _(label_distribution, mo):
    _counts = dict(label_distribution.iter_rows())
    _largest = max(_counts.values())
    _smallest = min(_counts.values())

    mo.md(f"""
    ### O que isso significa

    As classes têm **{_counts['low']} registros em `low`**,
    **{_counts['medium']} em `medium`** e **{_counts['high']} em `high`**. A maior
    classe é {_largest / _smallest:.2f} vez a menor.

    O desequilíbrio é moderado, não extremo. Ainda assim, o próximo experimento deve
    usar split estratificado e avaliar Macro-F1, recall de `high` e erros ordinais;
    acurácia isolada não será suficiente.
    """)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. O tamanho do texto muda conforme a classe?

    Comprimento pode virar um atalho: o modelo poderia associar textos maiores a uma
    classe sem aprender sinais de urgência. O boxplot permite comparar mediana,
    dispersão e outliers diretamente.
    """)
    return


@app.cell
def _(alt, dataset, pl):
    text_lengths = dataset.select(
        "urgency",
        pl.col("patient_text_en").str.len_chars().alias("characters"),
    )

    length_chart = (
        alt.Chart(text_lengths)
        .mark_boxplot(size=52)
        .encode(
            x=alt.X("urgency:N", sort=["low", "medium", "high"], title="Urgência"),
            y=alt.Y("characters:Q", title="Caracteres"),
            color=alt.Color("urgency:N", legend=None),
        )
        .properties(title="Comprimento do texto por classe", width=460, height=340)
        .interactive(name="text-length-zoom")
    )

    length_chart

    return (text_lengths,)


@app.cell(hide_code=True)
def _(mo, pl, text_lengths):
    _length_summary = text_lengths.group_by("urgency").agg(
        pl.col("characters").median().alias("median")
    ).sort("urgency")

    mo.md(f"""
    ### O que isso significa

    As medianas são próximas: **{_length_summary.filter(pl.col('urgency') == 'low').item(0, 'median'):.0f}**
    caracteres em `low`, **{_length_summary.filter(pl.col('urgency') == 'medium').item(0, 'median'):.0f}**
    em `medium` e **{_length_summary.filter(pl.col('urgency') == 'high').item(0, 'median'):.0f}**
    em `high`.

    Não há indicação visual de que comprimento, sozinho, separe claramente as
    classes. Ele continuará sendo monitorado, mas não será usado como justificativa
    para limpeza ou truncamento no baseline TF-IDF.
    """)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Há textos muito parecidos capazes de inflar a avaliação?

    Textos idênticos já foram descartados como problema. Agora representamos cada
    texto por TF-IDF de caracteres e medimos sua similaridade com o vizinho mais
    próximo. Valores acima de 0,90 são candidatos a quase duplicata; valores entre
    0,80 e 0,89 podem indicar templates compartilhados.
    """)
    return


@app.cell
def _(NearestNeighbors, TfidfVectorizer, alt, dataset, pl):
    _texts = dataset.get_column("patient_text_en").to_list()
    _vectors = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2
    ).fit_transform(_texts)
    _distances, _ = NearestNeighbors(
        n_neighbors=2, metric="cosine", n_jobs=-1
    ).fit(_vectors).kneighbors(_vectors)

    nearest_similarity = pl.DataFrame(
        {"similarity": (1 - _distances[:, 1]).tolist()}
    )
    similarity_bands = (
        nearest_similarity.with_columns(
            pl.when(pl.col("similarity") >= 0.90)
            .then(pl.lit(">= 0.90"))
            .when(pl.col("similarity") >= 0.80)
            .then(pl.lit("0.80-0.89"))
            .otherwise(pl.lit("< 0.80"))
            .alias("band")
        )
        .group_by("band")
        .len(name="records")
    )

    similarity_chart = (
        alt.Chart(alt.Data(values=nearest_similarity.to_dicts()))
        .mark_bar(color="#8b5cf6", opacity=0.9)
        .encode(
            x=alt.X(
                "similarity:Q",
                bin=alt.Bin(step=0.05),
                scale=alt.Scale(domain=[0, 1]),
                title="Similaridade com o texto mais próximo",
            ),
            y=alt.Y("count():Q", title="Textos"),
            tooltip=[alt.Tooltip("count():Q", title="Textos")],
        )
        .properties(title="Distribuição da similaridade entre textos", width=520, height=300)
        .interactive(name="near-duplicate-zoom", bind_x=True, bind_y=False)
    )

    similarity_chart

    return (similarity_bands,)


@app.cell(hide_code=True)
def _(dataset, mo, similarity_bands):
    _similarity_counts = dict(similarity_bands.iter_rows())
    _template_candidates = _similarity_counts.get("0.80-0.89", 0)
    _near_duplicates = _similarity_counts.get(">= 0.90", 0)
    _exact_duplicates = dataset.height - dataset.get_column("patient_text_en").n_unique()

    mo.md(f"""
    ### O que isso significa

    Há **{_exact_duplicates} duplicatas exatas**, **{_near_duplicates} textos com
    vizinho de similaridade ≥ 0,90** e **{_template_candidates} na faixa 0,80–0,89**.

    As duas últimas contagens representam documentos candidatos, não pares únicos e
    nem confirmação clínica de cópia. Candidatos acima de 0,90 precisam ser agrupados
    ou revisados antes de selar o split final; a faixa intermediária serve para
    investigar templates.
    """)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. As palavras dos rótulos entregam a classe?

    Os 112 registros encontrados anteriormente contêm `low`, `medium` ou `high`.
    Para saber se isso parece um atalho, comparamos a palavra encontrada com o target
    real. Uma concentração perfeita na diagonal seria um sinal forte de vazamento.
    """)
    return


@app.cell
def _(alt, dataset, pl):
    _term_rows = []
    for _term in ("low", "medium", "high"):
        _counts = (
            dataset.filter(pl.col("patient_text_en").str.contains(rf"(?i)\b{_term}\b"))
            .group_by("urgency")
            .len(name="records")
        )
        _found = dict(_counts.iter_rows())
        for _urgency in ("low", "medium", "high"):
            _term_rows.append(
                {"term": _term, "urgency": _urgency, "records": _found.get(_urgency, 0)}
            )

    term_target_counts = pl.DataFrame(_term_rows)
    _term_data = alt.Data(values=term_target_counts.to_dicts())
    _term_base = alt.Chart(_term_data).encode(
        x=alt.X("term:N", sort=["low", "medium", "high"], title="Palavra encontrada"),
        xOffset=alt.XOffset("urgency:N", sort=["low", "medium", "high"]),
        y=alt.Y("records:Q", title="Registros"),
        color=alt.Color("urgency:N", sort=["low", "medium", "high"], title="Target real"),
        tooltip=["term:N", "urgency:N", "records:Q"],
    )

    term_target_chart = (
        _term_base.mark_bar()
        + _term_base.mark_text(dy=-7).encode(text="records:Q")
    ).properties(title="Palavra encontrada versus target real", width=500, height=300).interactive(
        name="label-term-zoom"
    )

    term_target_chart

    return (term_target_counts,)


@app.cell(hide_code=True)
def _(mo, pl, term_target_counts):
    _matched_target = term_target_counts.filter(
        pl.col("term") == pl.col("urgency")
    ).get_column("records").sum()
    _term_total = term_target_counts.get_column("records").sum()

    mo.md(f"""
    ### O que isso significa

    Dos **{_term_total} registros** com palavras iguais aos rótulos,
    **{_matched_target} ({_matched_target / _term_total:.1%})** têm palavra e target
    coincidentes. Os demais aparecem em outra classe.

    As palavras não entregam deterministicamente o target, mas a associação não pode
    ser ignorada. Antes do treino, esses registros devem ser revisados de forma
    controlada, sem serem exibidos no notebook, e o desempenho deve ser comparado com
    e sem esses termos.
    """)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6. A prevalência muda entre faixas etárias?

    Com instituição e tempo ausentes, faixa etária é um dos poucos recortes
    disponíveis. Para evitar conclusões a partir de grupos minúsculos, mostramos
    somente faixas com pelo menos 20 registros e comparamos proporções, não contagens.
    """)
    return


@app.cell
def _(alt, dataset, pl):
    age_distribution = (
        dataset.group_by(["age_group", "urgency"])
        .len(name="records")
        .with_columns(pl.col("records").sum().over("age_group").alias("group_total"))
        .filter(pl.col("group_total") >= 20)
        .with_columns((pl.col("records") / pl.col("group_total")).alias("share"))
    )

    age_chart = (
        alt.Chart(age_distribution)
        .mark_bar()
        .encode(
            y=alt.Y("age_group:N", title="Faixa etária"),
            x=alt.X("share:Q", stack="normalize", axis=alt.Axis(format="%"), title="Proporção"),
            color=alt.Color("urgency:N", sort=["low", "medium", "high"], title="Urgência"),
            tooltip=["age_group:N", "urgency:N", alt.Tooltip("share:Q", format=".1%")],
        )
        .properties(title="Composição das classes por faixa etária", width=560, height=300)
        .interactive(name="age-group-zoom")
    )

    age_chart

    return (age_distribution,)


@app.cell(hide_code=True)
def _(age_distribution, mo, pl):
    _high_share = age_distribution.filter(pl.col("urgency") == "high")
    _highest = _high_share.sort("share", descending=True).row(0, named=True)
    _lowest = _high_share.sort("share").row(0, named=True)

    mo.md(f"""
    ### O que isso significa

    Entre os grupos com tamanho mínimo, a proporção de `high` varia de
    **{_lowest['share']:.1%} em `{_lowest['age_group']}`** a
    **{_highest['share']:.1%} em `{_highest['age_group']}`**.

    É uma diferença descritiva, não causal. Faixa etária pode apoiar auditoria de
    subgrupos, mas não deve entrar no modelo enquanto sua disponibilidade no momento
    zero não estiver documentada.
    """)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 7. Como reservar treino, validação e teste?

    Sem paciente e tempo, o melhor fallback disponível é uma divisão aleatória,
    estratificada e reproduzível. Usamos 70% para treino, 15% para validação e 15%
    para teste, com `seed=42`. O teste fica reservado para a avaliação final.
    """)
    return


@app.cell
def _(alt, dataset, pl, train_test_split):
    _indices = list(range(dataset.height))
    _targets = dataset.get_column("urgency").to_list()
    _development, _test = train_test_split(
        _indices, test_size=0.15, random_state=42, stratify=_targets
    )
    _development_targets = [_targets[index] for index in _development]
    _train, _validation = train_test_split(
        _development,
        test_size=3 / 17,
        random_state=42,
        stratify=_development_targets,
    )
    split_manifest = pl.concat(
        [
            pl.DataFrame({"row_index": _train, "split": "train"}),
            pl.DataFrame({"row_index": _validation, "split": "validation"}),
            pl.DataFrame({"row_index": _test, "split": "test"}),
        ]
    ).join(
        dataset.with_row_index().select("index", "id", "urgency"),
        left_on="row_index",
        right_on="index",
    )
    split_distribution = split_manifest.group_by(["split", "urgency"]).len(name="records")
    _split_data = alt.Data(values=split_distribution.to_dicts())
    _split_base = alt.Chart(_split_data).encode(
        x=alt.X("split:N", sort=["train", "validation", "test"], title="Conjunto"),
        xOffset=alt.XOffset("urgency:N", sort=["low", "medium", "high"]),
        y=alt.Y("records:Q", title="Registros"),
        color=alt.Color("urgency:N", sort=["low", "medium", "high"], title="Urgência"),
        tooltip=["split:N", "urgency:N", "records:Q"],
    )

    split_chart = (
        _split_base.mark_bar()
        + _split_base.mark_text(dy=-7).encode(text="records:Q")
    ).properties(title="Classes preservadas em cada conjunto", width=520, height=300).interactive(
        name="split-distribution-zoom"
    )

    split_chart

    return (split_manifest,)


@app.cell(hide_code=True)
def _(mo, split_manifest):
    _split_sizes = dict(
        split_manifest.group_by("split").len(name="records").iter_rows()
    )

    mo.md(f"""
    ### O que isso significa

    O manifesto reproduzível contém **{_split_sizes['train']} registros de treino**,
    **{_split_sizes['validation']} de validação** e **{_split_sizes['test']} de teste**.
    As proporções das classes foram preservadas.

    O teste está definido e não deve orientar limpeza, vocabulário, TF-IDF ou
    hiperparâmetros. Esta divisão é um fallback metodológico: sem `patient_id` e
    `timestamp`, não é possível garantir independência por paciente nem uma
    avaliação temporal realista.
    """)

    return


@app.cell(hide_code=True)
def _(mo, pl, similarity_bands, term_target_counts):
    _matched_target = term_target_counts.filter(
        pl.col("term") == pl.col("urgency")
    ).get_column("records").sum()
    _near_counts = dict(similarity_bands.iter_rows())
    _near = _near_counts.get(">= 0.90", 0)
    _template = _near_counts.get("0.80-0.89", 0)

    mo.md(f"""
    ## 8. Decisão

    ### B — pronto para modelagem somente após revisão dos candidatos semelhantes

    Os quatro pontos foram tratados:

    1. os 112 registros com palavras dos rótulos foram auditados por associação com
       o target; {_matched_target} coincidem com a classe publicada;
    2. foram encontrados {_near} textos com similaridade ≥ 0,90 e {_template} na
       faixa de possíveis templates;
    3. está registrado que a base não permite split por paciente ou tempo;
    4. foi criado um split estratificado e determinístico 70/15/15, com teste
       reservado.

    Antes do treino, os candidatos de similaridade ≥ 0,90 precisam ser revisados e,
    se confirmados como cópias, mantidos no mesmo conjunto.
    """)

    return


if __name__ == "__main__":
    app.run()
