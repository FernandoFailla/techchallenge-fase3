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

Os hooks locais executam `ruff check`, `ruff format --check`, `mypy` e `pytest` antes de cada commit. Para executar a mesma validação manualmente em todo o repositório, use `uv run pre-commit run --all-files`.

## Exploração

Após recuperar os dados com `make pull-data`, execute:

```bash
uv run marimo edit notebooks/02_exploratory_data_analysis.py
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
