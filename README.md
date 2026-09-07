# Tech Challenge - Fase 3

Protótipo educacional de NLP que recebe uma queixa textual em inglês e a
classifica como `normal`, `atencao` ou `urgente`. O repositório demonstra o
ciclo de vida completo de um modelo: dados versionados, treinamento
orquestrado, rastreabilidade, API, observabilidade, otimização e CI.

> **Aviso importante:** isto não é um produto clínico. Não use para diagnóstico,
> atendimento, priorização ou qualquer decisão médica real.

## Comece Aqui

Se esta é sua primeira vez no projeto, siga esta sequência sem pular etapas:

1. Instale os requisitos descritos em [Pré-requisitos](#pré-requisitos).
2. Clone o repositório, copie `.env.example` para `.env` e configure o acesso ao
   remote DVC, conforme [Preparar a máquina](#1-preparar-a-máquina).
3. Recupere os dados versionados com `make pull-data`.
4. Inicie MLflow e Airflow, execute a DAG `training_pipeline` uma vez para criar
   o primeiro modelo `champion`.
5. Inicie API, Prometheus e Grafana com `make observability`.
6. Verifique `/health`, faça uma chamada de contrato a `/predict` e abra o
   dashboard.

O primeiro bootstrap demora mais porque treina, avalia, converte e registra um
modelo. Depois disso, os volumes Docker preservam o MLflow e basta iniciar a
stack de observabilidade novamente.

## O Que Cada Componente Faz

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

| Componente | Papel | Onde acessar |
|---|---|---|
| DVC | Recupera a versão aprovada de dados e artefatos | Linha de comando |
| Airflow | Executa o pipeline manual de preparação e treinamento | `http://localhost:8080` |
| MLflow | Guarda experimentos, modelos e o alias `champion` | `http://localhost:5000` |
| FastAPI | Serve o modelo promovido | `http://localhost:8000` |
| Prometheus | Coleta métricas da API | `http://localhost:9090` |
| Grafana | Mostra requisições, p95 e taxa de erros | `http://localhost:3000` |

Airflow é iniciado separadamente da stack de serving. A API carrega uma única
versão do modelo no startup. Para aplicar outra versão, altere o alias no MLflow
e reinicie a API; não existe troca silenciosa de modelo em runtime.

## Pré-requisitos

- Git.
- Python 3.13 ou superior.
- [uv](https://docs.astral.sh/uv/).
- Docker Engine com Docker Compose v2 e permissão para executar `docker` sem
  `sudo`.
- Acesso à Internet.
- Credenciais OAuth do Google Drive compartilhadas pelo mantenedor, para a
  reprodução exata via DVC.

Confirme os itens principais antes de continuar:

```bash
git --version
python3 --version
uv --version
docker compose version
```

## Execução Do Zero

### 1. Preparar a máquina

Clone o projeto, instale as dependências bloqueadas no lockfile e crie seu
arquivo local de ambiente. Nunca versione `.env` nem compartilhe suas
credenciais.

```bash
git clone https://github.com/FernandoFailla/techchallenge-fase3.git
cd techchallenge-fase3
cp .env.example .env
uv sync --all-groups
uv run pre-commit install
```

Edite `.env` e preencha apenas `GDRIVE_CLIENT_ID` e `GDRIVE_CLIENT_SECRET` com
as credenciais recebidas do mantenedor. Os campos `MLFLOW_*` podem permanecer
com os valores de exemplo. Os comandos `make pull-data` e `make dvc-reauth`
carregam `.env` automaticamente; nao e necessario exportar as variaveis antes de
executa-los.

### Acesso ao Google Drive: responsabilidades

O projeto usa duas camadas distintas de acesso. Não confunda uma com a outra:

| Papel | Responsabilidade |
|---|---|
| Administrador | Mantém o projeto OAuth no Google Cloud, habilita a Google Drive API, fornece a configuração OAuth por canal privado e compartilha a pasta DVC com as contas autorizadas. |
| Novo usuário | Insere a configuração OAuth localmente e autoriza **a própria conta Google** no navegador quando o DVC solicitar. |

O novo usuário deve receber do administrador somente os valores de configuração
`GDRIVE_CLIENT_ID` e `GDRIVE_CLIENT_SECRET`. Ele os adiciona ao seu `.env`,
carrega o arquivo no terminal e executa `make pull-data`. Quando o navegador
abrir, deve entrar com a conta Google que o administrador adicionou à pasta DVC
e aceitar a autorização. O token gerado pertence àquela pessoa, fica no cache
local do sistema e **nunca deve ser enviado a outra pessoa**.

O administrador deve compartilhar a pasta Drive por e-mail ou grupo específico,
nunca por acesso público de link. Para este repositório, conceda `Visualizador`
ao usuário que só executará `make pull-data`; conceda `Editor` apenas a quem foi
explicitamente autorizado a publicar dados com `dvc push`. O fluxo normal do MVP
é somente leitura: DAGs e usuários não devem publicar no remote DVC.

O administrador deve criar ou manter um cliente OAuth do tipo **Desktop app**,
habilitar a Google Drive API e adicionar os novos usuários à tela de
consentimento quando o aplicativo estiver em modo de teste. A configuração do
cliente não autoriza acesso à pasta por si só: a autorização real depende da
conta Google do usuário e das permissões concedidas a ela na pasta.

Nunca versione `.env`, `.dvc/config.local` ou o cache de token OAuth. O projeto
já ignora esses caminhos. Para remover o acesso de uma pessoa, remova-a do
compartilhamento da pasta Drive; se for necessário invalidar todos os clientes,
o administrador deve rotacionar o cliente OAuth no Google Cloud e distribuir a
nova configuração por canal privado.

### 2. Recuperar os dados aprovados

O DVC é o caminho reprodutível. Ele recupera a versão dos dados associada ao
commit atual, incluindo a base de modelagem usada no treinamento.

```bash
make pull-data
uv run dvc status
```

Na primeira execução, o DVC pode abrir o navegador para o OAuth. Autorize a
conta que possui acesso à pasta Google Drive compartilhada e retorne ao
terminal. O comando deve terminar sem arquivos modificados.

`make download-data` é apenas uma alternativa para obter a fonte pública via
`KAGGLE_API_TOKEN`. Ele não substitui `make pull-data` para reproduzir a base
aprovada ou o modelo deste repositório.

### 3. Criar o primeiro modelo champion

Abra **dois terminais** no diretório do projeto. Mantenha os comandos abaixo em
execução; `Ctrl+C` encerra o serviço daquele terminal.

No terminal 1, inicie o MLflow:

```bash
make mlflow
```

No terminal 2, inicie o Airflow:

```bash
make airflow
```

O MLflow precisa estar em execução antes de disparar a DAG. O Airflow acessa o
serviço do host por `http://host.docker.internal:5000` dentro do container; não
defina `AIRFLOW_MLFLOW_TRACKING_URI` em `.env` na configuração local padrão.

Quando o Airflow estiver disponível, descubra a senha local gerada:

```bash
make airflow-password
```

Abra `http://localhost:8080`, entre com o usuário `admin` e a senha exibida.
Execute manualmente a DAG `training_pipeline` e aguarde as quatro tarefas:

```text
validate_modeling_base
  -> train_and_evaluate
  -> optimize_and_benchmark
  -> register_and_promote
```

A última tarefa cria uma versão de `triage-urgency-classifier` e associa o
alias `champion`. Você pode acompanhar runs, métricas e artefatos em
`http://localhost:5000`.

Também existe a DAG `prepare_modeling_base`, usada para materializar a base
determinística a partir do CSV bruto. Ela escreve somente no ambiente local: não
publica alterações no DVC, Google Drive ou Git.

### 4. Iniciar a API e a observabilidade

Depois que a DAG tiver promovido um `champion`, abra um terceiro terminal e
inicie a stack completa:

```bash
make observability
```

Espere os serviços terminarem o startup e confirme que a API carregou o modelo:

```bash
curl --fail http://localhost:8000/health
```

Uma resposta de sucesso contém `status: "ok"` e a versão do modelo carregado.
Se a API não iniciar, verifique primeiro se a DAG foi concluída e se o alias
`champion` existe no MLflow. A API falha intencionalmente em vez de escolher um
fallback.

### 5. Testar a API e o dashboard

Faça uma chamada de contrato com um texto neutro de demonstração. Não envie
dados pessoais, clínicos ou identificáveis.

```bash
curl --silent --show-error --fail \
  --request POST http://localhost:8000/predict \
  --header 'Content-Type: application/json' \
  --data '{"text":"Sample input for an API contract check."}'
```

A resposta possui o formato abaixo; a classificação depende do modelo ativo:

```json
{
  "classification": "normal|atencao|urgente",
  "model_version": "<versao-registry>"
}
```

Abra Grafana em `http://localhost:3000`. O login inicial é `admin` / `admin`;
altere a senha local quando solicitado. O dashboard provisionado
`Observabilidade da API` mostra total de inferências, latência p95 e taxa de
erros 5xx. O Prometheus coleta `/metrics` a cada 15 segundos, portanto espere
um intervalo de coleta após gerar tráfego.

## Comandos Mais Úteis

| Objetivo | Comando |
|---|---|
| Mostrar todos os comandos | `make help` |
| Executar lint, formato, tipos e testes | `make check` |
| Validar o Compose sem iniciar containers | `make docker-config` |
| Iniciar somente MLflow | `make mlflow` |
| Encerrar MLflow | `make mlflow-down` |
| Iniciar Airflow | `make airflow` |
| Mostrar senha do Airflow em execução | `make airflow-password` |
| Encerrar Airflow | `make airflow-down` |
| Apagar containers e volume local do Airflow | `make airflow-reset` |
| Construir somente a imagem da API | `make api-build` |
| Iniciar API, MLflow, Prometheus e Grafana | `make observability` |
| Encerrar a stack de observabilidade | `make observability-down` |
| Medir a API local | `make api-benchmark` |
| Refazer o login Google Drive e recuperar os dados | `make dvc-reauth` |

`make observability-down` remove containers, mas preserva os volumes de MLflow,
Prometheus e Grafana. `make airflow-reset` é destrutivo apenas para o estado
local do Airflow e exige que você execute as DAGs novamente.

## Desenvolvimento E Validação

Antes de abrir uma alteração, execute:

```bash
make check
make docker-config
```

Os notebooks Marimo são exploratórios. Abra-os com `uv run marimo edit` e, ao
final da edição, valide o arquivo:

```bash
uv run marimo edit notebooks/04_baseline_nlp.py
uv run marimo check notebooks/04_baseline_nlp.py
```

O GitHub Actions também executa `make check` e constrói `Dockerfile.api` em todo
push e pull request. Ele não baixa dados, treina ou promove modelos porque esses
passos exigem credenciais e têm custo maior.

## Troubleshooting

| Sintoma | Verificação e ação |
|---|---|
| `make pull-data` pede credenciais | Confirme que `.env` foi carregado e que a conta OAuth tem acesso ao Drive compartilhado. |
| `invalid_grant`, token expirado ou revogado | Execute `make dvc-reauth`, entre novamente com sua conta Google no navegador e conclua o consentimento. Esse comando remove somente o token OAuth local deste projeto; não apaga dados, volumes Docker nem arquivos do remote DVC. |
| A API não sobe | Abra MLflow, confirme o alias `champion` e execute novamente `make observability`. |
| Airflow não aceita a senha | Obtenha a senha atual com `make airflow-password`; após `make airflow-reset`, uma nova senha é criada. |
| A porta já está em uso | Encerre a stack correspondente com `make airflow-down` ou `make observability-down` e tente novamente. |
| Containers precisam de detalhes | Use `docker compose -f compose.mlflow.yml logs` ou `docker compose -f compose.airflow.yml logs`. Nunca publique logs que contenham dados sensíveis. |
| É necessário recomeçar o Airflow | Use `make airflow-reset`, suba-o novamente e reexecute a DAG. |

## Resultados Medidos

Snapshot executado nesta cópia em 23/08/2026, Linux x86_64 com `nproc=16`,
dataset DVC `data/processed/modeling_base.parquet` (2.000 registros; ponteiro
MD5 `023d3b5fe03abd10ddd21aa84f8baa00`). São medições locais de uma execução,
não representam SLO, capacidade de produção ou garantia de repetição em outra
máquina.

| Etapa | Medição observada | Protocolo da execução |
|---|---:|---|
| Baseline | Dummy Macro-F1 validação `0.187`; TF-IDF + Random Forest `0.678` | seleção na validação; Random Forest selecionado |
| Teste reservado | Macro-F1 `0.758`; acurácia `0.756` | modelo selecionado, 299 registros de teste |
| ONNX | scikit-learn `36.529 ms`; ONNX Runtime `0.417 ms`; `87.686x` | média por texto; 64 referências de validação, 1 warm-up e 5 repetições |
| Paridade ONNX | `1.000` | 299 previsões do teste reservado iguais às do scikit-learn |
| HTTP local | média `2.653 ms`; P50 `2.612 ms`; P95 `2.998 ms` | 20 warm-ups e 200 `POST /predict` sequenciais; API Docker; modelo MLflow versão `5` |

O benchmark HTTP registra somente agregados no experimento MLflow
`kan-24-http-benchmark`. O benchmark ONNX atual mede médias, com 1 warm-up e 5
repetições; portanto, ainda não comprova o protocolo planejado de 20 warm-ups,
500 predições individuais, mediana e p95. Reexecute os benchmarks na máquina de
avaliação antes de alegar esse gate como atendido.

## Dados, Segurança E Limitações

A fonte é [KurMed-Triage v1 no Kaggle](https://www.kaggle.com/datasets/alanjafari/kurmed-triage),
atribuída a Alan Jafari, com 2.000 exemplos sintéticos em inglês e target
publicado `urgency` (`low`, `medium`, `high`). A API apresenta esses rótulos como
`normal`, `atencao` e `urgente`; não cria nem altera a taxonomia.

Os metadados públicos consultados divergem entre CC BY 4.0 e CC BY-SA 4.0. Até
a confirmação por um arquivo de licença inequívoco da versão usada, o projeto
adota a interpretação mais restritiva, CC BY-SA 4.0, e mantém a atribuição.

- Não envie texto clínico, dados pessoais ou identificadores ao serviço.
- A aplicação não persiste texto submetido em logs, métricas ou MLflow.
- A stack local não inclui autenticação, autorização, alta disponibilidade,
  escalabilidade comprovada, monitoramento de drift ou retreinamento automático.
- O corpus é sintético e pequeno; métricas não demonstram validade clínica,
  segurança, generalização ou justiça entre populações.

## Documentação Complementar

- [Decisão teórica de cloud](docs/decisao-cloud.md)
- [Roteiro e checklist STAR](docs/roteiro-video-star.md)
- [ADR de dados](docs/adr/0001-dominio-e-dataset.md)
- [ADR de arquitetura](docs/adr/0002-arquitetura-do-mvp.md)
- [Plano e gates do MVP](docs/plano-mvp.md)
