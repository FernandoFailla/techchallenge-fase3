# Tech Challenge - Fase 3

Projeto da terceira fase da pós-graduação Machine Learning Engineering da Fiap.

O objetivo é aplicar NLP à classificação de urgência de baseado em queixa medicas, com o foco em demonstrar, o ciclo de vida completo de MLOps: validação de dados, treinamento, avaliação, otimização, registro, disponibilização por API, orquestração, CI/CD e monitoramento.

> Este sistema não possui validade clínica. Não deve ser usado para diagnóstico, atendimento ou tomada de decisão médica real.

## Estrutura inicial

```text
.
├── docs/                 # Documentação, ADRs e glossário
├── notebooks/            # Exploração e validação em Marimo
├── src/techchallenge/    # Código-fonte reutilizável do projeto
├── tests/                # Testes automatizados do código em src/
├── AGENTS.md             # Regras permanentes de desenvolvimento
└── pyproject.toml        # Metadados e configuração das ferramentas Python
```

## Ambiente de desenvolvimento

O projeto requer Python 3.13 e utiliza [uv](https://docs.astral.sh/uv/) para gerenciar dependências e executar comandos.

```bash
uv sync --all-groups
uv run pre-commit install
```

Os hooks locais executam `ruff check`, `ruff format --check`, `mypy` e `pytest` antes de cada commit. Para executar a mesma validação manualmente em todo o repositório, use `make check`.

Para iniciar o Airflow local que materializa a base de modelagem, use `make airflow`. A DAG não publica dados no DVC; após revisar os artefatos gerados, a decisão de versioná-los é manual. Consulte a senha local gerada com `make airflow-password`.

Se uma alteração de configuração ou permissões deixar o ambiente local inconsistente, execute `make airflow-reset` e inicie-o novamente. Esse comando remove apenas o estado local do Airflow.

Para iniciar o MLflow Tracking local, use `make mlflow`. O servidor fica disponível em `http://localhost:5000` e persiste seu banco SQLite e artefatos em volume Docker.

## API de inferência

A API FastAPI é executada em um contêiner não-root e não inclui dados, artefatos de
modelo ou credenciais na imagem. Ela recupera o modelo pelo MLflow no startup, logo é
necessário promover o alias `champion` antes de iniciá-la. O Compose mantém a API em um
profile separado para permitir o bootstrap do MLflow e o treinamento inicial.

```bash
# Primeiro bootstrap: inicia somente o MLflow, executa a DAG e promove o champion.
make mlflow
make airflow

# Depois que o champion existir, inicia MLflow e API.
make api
```

A API fica disponível em `http://localhost:8000`. O healthcheck só fica saudável após
o carregamento do modelo e pode ser consultado sem enviar texto clínico:

```bash
curl http://localhost:8000/health
```

Com a API e o MLflow locais em execução, execute o benchmark HTTP sequencial com:

```bash
make api-benchmark
```

O benchmark usa `http://localhost:8000`, descarta 20 requisições de aquecimento e mede
200 chamadas `POST /predict` sequenciais. A entrada é sintética e determinística, fica
somente em memória e não é enviada ao MLflow. O experimento `kan-24-http-benchmark`
registra a versão do modelo, parâmetros de execução e latências agregadas (média, P50 e
P95). Para explorar o resultado em Marimo, com o contêiner já saudável, use:

```bash
uv run marimo edit notebooks/07_http_api_benchmark.py
```

O serviço `api` usa `MLFLOW_TRACKING_URI=http://mlflow:5000` internamente. Para
selecionar outro modelo promovido, defina `MLFLOW_MODEL_URI` com uma URI de Registry
determinística antes de `make api`, por exemplo
`models:/triage-urgency-classifier@champion` ou
`models:/triage-urgency-classifier/3`. A imagem é construída com `uv.lock` e copia
somente `pyproject.toml`, `uv.lock`, `README.md` e `src/`; `.dockerignore` exclui dados
e arquivos de ambiente do contexto de build. Valide o Compose com `make docker-config`
e construa apenas a imagem com `make api-build`.

A DAG manual `training_pipeline` executa KAN-11, KAN-10 e KAN-13 uma única vez,
na ordem de validação, treinamento, benchmark e registro. Inicie `make mlflow`
antes de `make airflow`; o contêiner do Airflow usa `host.docker.internal:5000` por
defeito, ou respeita `MLFLOW_TRACKING_URI`. Cada execução cria novos runs e uma nova
versão no Model Registry; o alias `champion` só muda depois da verificação do bundle.

## Exploração

Após recuperar os dados com `make pull-data`, execute:

```bash
uv run marimo edit notebooks/02_exploratory_data_analysis.py
```

Para executar o baseline KAN-11, inicie o MLflow em outro terminal e abra o
notebook. Ele registra apenas métricas e artefatos agregados; textos, IDs e o
modelo treinado não são enviados ao tracking server.

```bash
make mlflow
uv run marimo edit notebooks/04_baseline_nlp.py
```

Para executar o benchmark ONNX do KAN-10, mantenha o MLflow iniciado e abra o
notebook abaixo. Ele seleciona o Random Forest pela validação do KAN-11, mede o
fluxo completo de texto para classe com referências determinísticas da validação
e usa o teste reservado apenas para a paridade de classes entre scikit-learn e
ONNX Runtime. Pruning por `ccp_alpha` é um fallback de validação quando o gate
de speedup não é atendido.

```bash
make mlflow
uv run marimo edit notebooks/05_onnx_benchmark.py
```

O KAN-13 reproduz o fluxo aprovado do KAN-10 e registra o PyFunc local
`triage-urgency-classifier`. O bundle contém o vetor TF-IDF serializado, o
classificador ONNX e o mapa de classes para inferência; o MLflow registra apenas
proveniência, métricas agregadas e hashes como metadados, sem expor vocabulário,
textos clínicos, IDs ou segredos. O alias `champion` só é atribuído após baixar
o bundle do Registry e validar todos os hashes.

```bash
make mlflow
uv run marimo edit notebooks/06_mlflow_model_registry.py
```

## Dataset

O protótipo utiliza o dataset público [KurMed-Triage v1 no Kaggle](https://www.kaggle.com/datasets/alanjafari/kurmed-triage). Seu uso está condicionado à validação de origem, licença, schema e qualidade definida no gate de dados.

Ele foi escolhido para o MVP por disponibilizar 2.000 exemplos sintéticos em inglês, uma coluna textual (`patient_text_en`) e três níveis de urgência já publicados (`low`, `medium` e `high`). Isso permite demonstrar o pipeline de NLP e MLOps sem criar, traduzir ou agregar rótulos localmente.

As bases sugeridas na ata foram o Medical Abstracts TC Corpus e recortes do MIMIC-III. O Medical Abstracts não foi adotado porque classifica especialidades, e não níveis de urgência. A ata não especifica um recorte ou target de urgência do MIMIC-III; por isso, para este MVP, foi priorizado o KurMed-Triage, que já publica três níveis de urgência diretamente compatíveis com o problema. A escolha não representa validade clínica: o dataset é sintético e permanece sujeito ao gate de dados e à verificação conservadora de licença.

### Sincronizar dados versionados

Após clonar o repositório, instale as dependências com `uv sync --all-groups`. Para acessar o dataset versionado, o mantenedor precisa compartilhar a pasta do Google Drive com sua conta e adicioná-la como usuária de teste do aplicativo OAuth do projeto.

Receba `GDRIVE_CLIENT_ID` e `GDRIVE_CLIENT_SECRET` do mantenedor por canal seguro e preencha seu `.env` local. Não versione esse arquivo nem envie essas credenciais para canais públicos. Em seguida, execute:

```bash
set -a
source .env
set +a
make pull-data
```

Na primeira execução, o navegador solicitará autorização para acessar a pasta compartilhada. Depois disso, basta executar `make pull-data` sempre que quiser recuperar a versão de dados associada ao commit atual.
