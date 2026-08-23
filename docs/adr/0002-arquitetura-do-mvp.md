# ADR 0002: Arquitetura do MVP

- Status: Aceita
- Data: 12/08/2026
- Decisores: Fernando Failla

## Contexto

O projeto deve demonstrar uma API FastAPI em Docker, pipeline de treino com Airflow, CI/CD, monitoramento com Prometheus e Grafana e uma tecnica de otimizacao de latencia. O prazo, a execucao individual e o carater educacional favorecem uma arquitetura local, reproduzivel e com poucos mecanismos operacionais.

MLflow Tracking e Model Registry e DVC foram adicionados ao escopo para tornar experimentos, modelos e dados rastreaveis. Esses componentes nao podem transformar o MVP em uma plataforma distribuida ou consumir a margem necessaria aos itens avaliados.

## Decisao

Adotar uma API de inferencia sincrona e stateless, com modelo carregado no startup, apoiada por servicos locais separados para registry e observabilidade.

### Componentes

| Componente | Responsabilidade |
|---|---|
| FastAPI | Validar requisicoes e servir classificacoes em tempo real |
| MLflow | Rastrear experimentos, armazenar artefatos e resolver a versao `champion` |
| SQLite | Persistir metadados do MLflow Model Registry |
| ONNX Runtime | Executar o classificador otimizado |
| Prometheus | Coletar contagem, latencia e erros da API |
| Grafana | Apresentar os tres paineis obrigatorios |
| Airflow standalone | Orquestrar validacao, treinamento e registro sob demanda |
| DVC | Versionar referencias a dados e artefatos finais |
| GitHub Actions | Executar qualidade, testes e build da imagem |

### Empacotamento do modelo

O modelo registrado `triage-urgency-classifier` sera um MLflow PyFunc com:

- `TfidfVectorizer` treinado e persistido;
- classificador convertido para ONNX;
- mapeamento da saida ONNX para as classes da API;
- dependencias e assinatura de entrada/saida;
- inicializacao da sessao ONNX Runtime em `load_context()`.

A assinatura recebera um DataFrame com uma coluna string `text` e uma ou mais linhas, e retornara um DataFrame com a coluna string `classification`, preservando a ordem do lote. A matriz esparsa produzida pelo TF-IDF sera convertida para o tensor denso `float32` `[batch, features]` esperado pelo classificador ONNX. A exportacao persistira os nomes dos tensores e a ordem de `classes_`; testes cobrirao paridade das tres classes e inferencia em lote.

A conversao ONNX sera aplicada inicialmente ao classificador, mantendo o TF-IDF do scikit-learn. Essa divisao reduz o risco de incompatibilidade na conversao do pre-processamento textual. O benchmark medira o fluxo completo de texto ate classe e uma chamada HTTP ao container para que o ganho relatado represente a experiencia real da API. O modelo original e o otimizado usarao a mesma amostra, maquina, worker e sequencia de execucao.

### Ciclo de promocao

1. `validate_data` valida a fonte fixada pelo DVC.
2. `train_and_evaluate` treina o baseline e o candidato e aplica o gate de qualidade.
3. `optimize_and_benchmark` exporta o mesmo candidato para ONNX, verifica equivalencia e mede desempenho; se necessario, seleciona pruning sem usar o teste e reaplica os gates ao novo candidato.
4. `register_and_promote` cria um bundle com os artefatos aprovados, registra seus hashes e run ID e, somente depois, associa `champion` a versao.
5. A API resolve `models:/triage-urgency-classifier@champion` no startup e informa essa versao nas respostas.
6. Uma troca de versao requer reassociar o alias e reiniciar a API.

Nao havera hot-swap, polling do registry ou endpoint administrativo no MVP. Se o alias nao existir ou o artefato nao puder ser carregado, a API falhara no startup em vez de servir um fallback silencioso.

### Processos locais

O Docker Compose incluira quatro servicos:

- API
- MLflow
- Prometheus
- Grafana

O MLflow usara SQLite como backend e um diretorio persistente como artifact store. O tracking server atuara como proxy dos artefatos para que Airflow e API usem URIs HTTP em vez de caminhos locais diferentes entre host e containers. O Airflow sera executado em um container standalone fora desse Compose para evitar banco, scheduler e webserver adicionais na stack demonstrada.

No primeiro bootstrap, somente o MLflow sera iniciado. A DAG criara e promovera a primeira versao; API, Prometheus e Grafana serao iniciados depois que `champion` puder ser resolvido. Reinicios posteriores reutilizarao os volumes persistentes.

### Contratos da API

- `POST /predict`: recebe o texto e retorna `classification` e `model_version`.
- `GET /health`: informa se o modelo foi carregado e qual versao esta ativa.
- `GET /metrics`: expoe metricas no formato Prometheus.

Probabilidades nao serao retornadas no MVP porque o Random Forest nao estara calibrado. Textos de entrada nao serao persistidos ou usados como labels de metricas.

### Observabilidade

As metricas minimas serao:

- total de requisicoes por endpoint, metodo e status;
- histograma de latencia por endpoint e metodo;
- total ou taxa de erros derivada de status.

Labels terao cardinalidade limitada e nunca conterao texto submetido, detalhes de excecao ou identificadores de paciente. O Grafana exibira total de requisicoes, p95 e taxa de erro.

### CI/CD

Em push e pull request, o GitHub Actions executara:

1. Ruff format/check.
2. mypy.
3. pytest.
4. build do Dockerfile da API.

O workflow nao treinara nem promovera modelos, pois isso aumentaria tempo, dependencia de dados e necessidade de credenciais. Treino e promocao permanecem demonstracoes locais orquestradas pelo Airflow.

### Estrategia de deploy

O padrao arquitetural escolhido e inferencia em tempo real com um container stateless. Deploy real em nuvem esta fora do MVP. O provedor e o servico gerenciado especificos serao comparados e documentados no README no card final, depois das aulas de cloud, sem alterar a arquitetura local nem implementar infraestrutura remota.

## Alternativas consideradas

### Servir diretamente com MLflow Models Serve

Rejeitada. A rubrica exige uma API FastAPI instrumentada e com contratos proprios.

### Registrar apenas o modelo scikit-learn

Rejeitada. O registry deixaria de representar o artefato otimizado realmente servido.

### Converter todo o pipeline de texto para ONNX desde o inicio

Adiada. Poderia reduzir dependencias de runtime, mas aumenta o risco de problemas com operadores de texto e a conversao do TF-IDF.

### Hot-swap do alias em runtime

Rejeitada. Exigiria concorrencia, consistencia de versoes, rollback e observabilidade adicionais sem beneficio para a rubrica.

### Incluir Airflow no Compose principal

Rejeitada no MVP. Aumentaria o numero de servicos e bancos necessarios para demonstrar a stack obrigatoria.

### PostgreSQL para MLflow

Rejeitada no MVP. SQLite atende a demonstracao local e habilita o Model Registry com menor custo operacional.

### Cloud real

Adiada para alem do MVP. A rubrica exige uma decisao textual, nao provisionamento.

## Consequencias

### Positivas

- O bundle registrado contem exatamente o TF-IDF e o ONNX aprovados no run e corresponde ao modelo servido.
- O alias desacopla a API de numeros de versao.
- Startup deterministico evita modelos trocados durante uma requisicao.
- A arquitetura cobre a rubrica com poucos servicos.
- Dados, experimentos e modelos possuem rastreabilidade independente.

### Negativas

- Uma promocao exige indisponibilidade breve durante o restart.
- SQLite e volumes locais nao oferecem alta disponibilidade.
- O runtime ainda depende de scikit-learn para o TF-IDF.
- Airflow fora do Compose exige um comando separado na demonstracao.
- O servico MLflow amplia a stack alem dos tres componentes obrigatorios.

## Invariantes

- Uma instancia da API serve uma unica versao durante todo seu ciclo de vida.
- Somente uma versao possui o alias `champion` por vez.
- A versao informada pela API corresponde ao artefato carregado, ao run ID e aos hashes registrados.
- Nenhuma requisicao e atendida antes do carregamento bem-sucedido do modelo.
- Texto medico nunca e emitido para logs, metricas ou MLflow.
- Uma versao nao recebe `champion` sem passar os gates pre-promocao definidos no plano; gates de servico e entrega sao verificados depois da promocao.

## Reavaliacao

Esta decisao deve ser revista se houver deploy real, multiplas replicas, promocao sem indisponibilidade, autenticacao, requisitos de alta disponibilidade ou migracao dos stores para servicos remotos.
