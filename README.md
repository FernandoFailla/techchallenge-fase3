# Tech Challenge - Fase 3

Protótipo educacional de NLP para classificar uma queixa textual em inglês como
`normal`, `atencao` ou `urgente`. Ele demonstra rastreabilidade de dados e
modelos, orquestração, API, observabilidade e otimização de inferência.

> Não tem validade clínica. Não use para diagnóstico, atendimento ou decisão
> médica real.

## Arquitetura

```text
KurMed-Triage v1 (fonte pública)
             |
     DVC + Google Drive remoto
             |
 data/raw -> data/processed
             |
 Airflow manual: validar -> treinar -> ONNX -> registrar/promover
             |                                  |
             +------------------------------> MLflow + SQLite + artefatos
                                                    |
                              triage-urgency-classifier@champion
                                                    |
 FastAPI (/predict, /health, /metrics) <- Prometheus <- Grafana
```

O Airflow é executado separadamente da stack de serving. A API é stateless,
carrega uma única versão `champion` no startup e não troca modelo em runtime.
O GitHub Actions executa `make check` e constrói a imagem da API em `push` e
pull request.

## Bootstrap Local

Requisitos: Docker Compose, Python 3.13, `uv` e acesso ao remote DVC
compartilhado. Copie `.env.example` para `.env`, preencha somente as credenciais
OAuth do Google Drive recebidas do mantenedor e não versione o arquivo.

```bash
uv sync --all-groups
uv run pre-commit install
set -a; source .env; set +a
make pull-data
```

`make pull-data` recupera os ponteiros DVC do commit atual. Na primeira
execução, o OAuth abre o navegador; a conta precisa ter acesso à pasta remota e
estar autorizada no aplicativo OAuth. `make download-data` é uma alternativa de
obtenção da fonte pública com `KAGGLE_API_TOKEN`; ele não substitui o estado
versionado recuperado por DVC.

Em terminais separados, inicialize e execute a primeira promoção:

```bash
# Terminal 1: MLflow em http://localhost:5000
make mlflow

# Terminal 2: Airflow em http://localhost:8080
make airflow

# Terminal 3, depois que o Airflow estiver saudável: senha local gerada
make airflow-password
```

No Airflow, execute manualmente a DAG `training_pipeline` e aguarde as etapas
`validate_modeling_base`, `train_and_evaluate`, `optimize_and_benchmark` e
`register_and_promote`. Ela cria e verifica o bundle antes de atribuir o alias
`champion`. Para reiniciar somente o estado local do Airflow, use
`make airflow-reset`; esse comando remove seu volume.

Depois de existir um `champion`, inicie a stack final:

```bash
make docker-config
make observability
curl --fail http://localhost:8000/health
```

Endpoints locais: MLflow `:5000`, API `:8000`, Prometheus `:9090` e Grafana
`:3000`. O primeiro acesso ao Grafana usa `admin`/`admin`; altere a senha local
quando solicitado. `make observability-down` remove os containers e preserva os
volumes. `make api-build` constrói somente a imagem e `make api-down` também
derruba a stack de serving.

O modelo servido pode ser escolhido antes de subir a API com
`MLFLOW_MODEL_URI`, por exemplo `models:/triage-urgency-classifier@champion` ou
uma versão explícita. A API falha no startup se não conseguir carregar o modelo;
não há fallback silencioso.

## Operação E Validação

```bash
make check                 # Ruff, mypy e pytest via pre-commit
make api-benchmark         # 20 warm-ups + 200 POSTs sequenciais locais
uv run marimo edit notebooks/04_baseline_nlp.py
uv run marimo edit notebooks/05_onnx_benchmark.py
uv run marimo edit notebooks/07_http_api_benchmark.py
```

O Prometheus coleta `/metrics` a cada 15 segundos. O dashboard provisionado
`Observabilidade da API` exibe total de inferências, p95 e taxa 5xx, filtrando
`/predict`. As métricas possuem somente `endpoint`, `method` e `status` como
labels; texto submetido, identificadores e detalhes de exceção não são
persistidos em logs, métricas ou MLflow.

## Resultado Medido

Snapshot executado nesta cópia em 23/08/2026, Linux x86_64 com `nproc=16`,
dataset DVC `data/processed/modeling_base.parquet` (2.000 registros; ponteiro
MD5 `023d3b5fe03abd10ddd21aa84f8baa00`). São medições locais de uma execução,
não SLO, capacidade de produção nem garantia de repetição em outra máquina.

| Etapa | Medição observada | Protocolo da execução |
|---|---:|---|
| Baseline | Dummy Macro-F1 validação `0.187`; TF-IDF + Random Forest `0.678` | seleção na validação; Random Forest selecionado |
| Teste reservado | Macro-F1 `0.758`; acurácia `0.756` | modelo selecionado, 299 registros de teste |
| ONNX | scikit-learn `36.529 ms`; ONNX Runtime `0.417 ms`; `87.686x` | média por texto; 64 referências de validação, 1 warm-up e 5 repetições |
| Paridade ONNX | `1.000` | 299 previsões do teste reservado iguais às do scikit-learn |
| HTTP local | média `2.653 ms`; P50 `2.612 ms`; P95 `2.998 ms` | 20 warm-ups e 200 `POST /predict` sequenciais; API Docker; modelo MLflow versão `5` |

O benchmark HTTP registra somente agregados no experimento MLflow
`kan-24-http-benchmark`. O benchmark ONNX atual mede médias, com 1 warm-up e 5
repetições; portanto ainda não demonstra o protocolo de aceite planejado de 20
warm-ups, 500 predições individuais, mediana e p95. Reexecute os notebooks ou a
DAG na máquina de avaliação e registre o novo snapshot antes de alegar esse
gate como atendido.

## Dados, Limitações E Decisão Cloud

A fonte é [KurMed-Triage v1 no Kaggle](https://www.kaggle.com/datasets/alanjafari/kurmed-triage),
atribuída a Alan Jafari, com 2.000 exemplos sintéticos em inglês e o target
publicado `urgency` (`low`, `medium`, `high`). A API apenas apresenta esses
rótulos como `normal`, `atencao` e `urgente`; não cria nem altera a taxonomia.

Os metadados públicos consultados divergem entre CC BY 4.0 e CC BY-SA 4.0. Até
uma licença inequívoca ser verificada no arquivo da versão usada, o projeto adota
a interpretação mais restritiva, CC BY-SA 4.0, mantém a atribuição e não afirma
uma licença definitiva. Consulte o [ADR de dados](docs/adr/0001-dominio-e-dataset.md)
para fonte, gate e limites completos.

O corpus é sintético, pequeno e não representa validação clínica, segurança,
generalização ou justiça entre populações. A stack local também não oferece
autenticação, alta disponibilidade, escalabilidade comprovada, monitoramento de
drift, retreino automático ou deploy em nuvem.

- [Decisão teórica de cloud](docs/decisao-cloud.md)
- [Roteiro e checklist STAR](docs/roteiro-video-star.md)
- [ADR de arquitetura](docs/adr/0002-arquitetura-do-mvp.md)
- [Plano e gates do MVP](docs/plano-mvp.md)
