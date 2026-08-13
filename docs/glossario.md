# Glossario

## Termos de dominio

### Atencao

Classe de apresentacao correspondente ao target publicado `medium`. Indica que o caso nao pertence aos extremos `normal` ou `urgente`. Nao constitui recomendacao medica.

### Classificacao

Rotulo textual retornado pela API: `normal`, `atencao` ou `urgente`.

### Queixa textual

Texto em ingles informado ao classificador. No dataset principal corresponde a `patient_text_en`.

### Normal

Classe de apresentacao correspondente ao target publicado `low`. Nao significa ausencia de doenca ou autorizacao para dispensar atendimento.

### Triagem

Classificacao educacional de prioridade a partir de texto. Neste projeto nao representa uma triagem clinica validada.

### Urgente

Classe de apresentacao correspondente ao target publicado `high`. E uma saida de modelo e nao substitui avaliacao profissional.

### Urgencia

Target do problema de classificacao. No KurMed-Triage e publicado como `low`, `medium` ou `high`.

## Dados e modelagem

### Checksum

Resumo criptografico usado para confirmar que um arquivo corresponde exatamente a versao validada.

### DVC

Data Version Control. Ferramenta que rastreia referencias a dados e artefatos grandes sem armazena-los diretamente no Git.

### Gate

Conjunto de criterios objetivos que precisa ser aprovado antes que um dado ou modelo avance para a proxima etapa.

### Majority baseline

Referencia que sempre preve a classe mais frequente. Sera implementada com `DummyClassifier` para verificar se o modelo principal aprende mais do que a distribuicao majoritaria.

### Macro-F1

Media nao ponderada do F1 de cada classe. Da o mesmo peso a `normal`, `atencao` e `urgente`, mesmo quando a distribuicao e desbalanceada.

### Matriz de confusao

Tabela que compara classes reais e preditas, permitindo identificar quais classes sao confundidas pelo modelo.

### Pruning

Reducao da complexidade das arvores. No Random Forest sera avaliada por `ccp_alpha` caso a conversao ONNX isolada nao demonstre ganho de latencia.

### Split estratificado

Separacao entre treino e teste que procura preservar a proporcao de cada classe.

### Target

Coluna que contem o rotulo a ser previsto. Neste projeto e `urgency`.

### TF-IDF

Representacao numerica que pondera termos conforme sua frequencia no documento e sua raridade no corpus.

## MLOps e servico

### Alias `champion`

Referencia mutavel do MLflow para a versao aprovada que deve ser carregada pela API.

### Artifact store

Local onde o MLflow persiste arquivos de modelos, vetorizadores, avaliacoes e outros artefatos associados aos runs.

### Backend store

Banco onde o MLflow persiste metadados de experimentos e do Model Registry. No MVP sera SQLite.

### CI/CD

Automacoes acionadas por alteracoes no repositorio. No MVP incluem lint, verificacao de tipos, testes e build Docker, mas nao deploy real.

### DAG

Directed Acyclic Graph. Definicao do Airflow que organiza as dependencias entre validacao, treinamento, otimizacao, registro e promocao do modelo.

### Hot-swap

Troca do modelo de uma API em execucao sem reinicio. Esta fora do escopo do MVP.

### MLflow Model Registry

Catalogo de modelos e versoes que permite registrar o PyFunc e associar o alias `champion` a uma versao aprovada.

### MLflow PyFunc

Formato de modelo com uma interface `predict` comum. No projeto encapsulara TF-IDF, ONNX Runtime e mapeamento de classes, recebendo lotes na coluna `text` e devolvendo a coluna `classification`.

### Model version

Numero monotonicamente crescente atribuido pelo MLflow a cada modelo registrado. A API retornara a versao efetivamente carregada.

### ONNX

Open Neural Network Exchange. Formato portavel usado para representar o classificador otimizado.

### ONNX Runtime

Motor que executa o artefato ONNX durante a inferencia.

### Promocao

Associacao do alias `champion` a uma versao que passou os gates pre-promocao. No MVP a promocao so entra em vigor na API depois de um restart.

### Run

Execucao rastreada pelo MLflow contendo parametros, metricas, tags e artefatos de um experimento.

### Startup

Fase de inicializacao da API em que o modelo `champion` e resolvido e carregado antes de aceitar requisicoes.

## Observabilidade e desempenho

### Cardinalidade

Quantidade de combinacoes possiveis dos valores de labels de uma metrica. Labels livres, como texto medico, sao proibidas por causarem alta cardinalidade e vazamento de dados.

### Latencia

Tempo entre o inicio e o fim de uma operacao. O benchmark medira o fluxo completo do texto ate a classe e tambem o tempo de resposta HTTP da API em Docker.

### Mediana

Percentil 50 das medicoes de latencia, menos sensivel a valores extremos que a media.

### p95

Percentil 95. Indica um valor abaixo do qual se encontram 95% das medicoes observadas.

### Prometheus

Sistema que coleta as metricas expostas pela API.

### Grafana

Ferramenta que consulta o Prometheus e apresenta os paineis de requisicoes, p95 e taxa de erro.

### Taxa de erro

Proporcao de requisicoes classificadas como erro em relacao ao total observado no periodo.

### Warm-up

Execucoes descartadas antes da coleta do benchmark para reduzir o efeito de inicializacao e caches frios.

## Entrega

### MVP

Minimum Viable Product. Menor entrega que cobre os requisitos obrigatorios e os gates definidos no plano.

### STAR

Estrutura do video final: Situation, Task, Action e Result.

### Conventional Commits

Convencao para mensagens de commit, como `docs:`, `feat:`, `fix:` e `test:`, usada para manter o historico semantico.
