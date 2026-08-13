# Plano do MVP

## 1. Objetivo

Entregar ate 15/09/2026 um prototipo educacional de triagem textual que classifique uma queixa em ingles como `normal`, `atencao` ou `urgente` e demonstre o ciclo de vida de um modelo de NLP: validacao de dados, treinamento, otimizacao, registro, inferencia via API, orquestracao, CI/CD e monitoramento.

O sistema nao possui validade clinica e nao deve ser usado para diagnostico, atendimento ou tomada de decisao medica real.

## 2. Restricoes

- Trabalho individual com disponibilidade aproximada de 5h30 por semana.
- Prazo final em 15/09/2026.
- Implementacao deliberadamente simples, priorizando os requisitos avaliados.
- Dataset publicado e previamente rotulado por terceiros.
- Nenhuma geracao, traducao, agregacao ou anotacao propria dos targets.
- Codigo, nomes de arquivos e identificadores em ingles.
- Documentacao e cards Jira em portugues brasileiro.
- Historico Git com Conventional Commits.
- Nenhum texto medico em logs, metricas, detalhes de erro ou rastreamento de experimentos.

## 3. Escopo do MVP

### 3.1 Dados

- Avaliar e, se aprovado, usar o KurMed-Triage v1.
- Consumir apenas `patient_text_en` como entrada e `urgency` como target.
- Mapear os rotulos publicados: `low` para `normal`, `medium` para `atencao` e `high` para `urgente`.
- Validar arquivo, origem, versao, checksum, licenca, schema, quantidade de linhas, classes, nulos, IDs duplicados e textos duplicados.
- Rastrear dados e artefatos finais com DVC.
- Usar Google Drive compartilhado como primeiro remote DVC e manter origem e checksum documentados como contingencia.

O KurMed-Triage somente sera aceito depois que o gate de dados for concluido. Caso falhe, o fallback devera ser outro corpus publicado, acessivel e previamente rotulado com exatamente tres classes de urgencia. Nao sera permitido criar os rotulos necessarios a partir de outra taxonomia.

### 3.2 Modelagem

- Criar um majority baseline com `DummyClassifier`.
- Treinar `TfidfVectorizer` com `RandomForestClassifier` como modelo principal.
- Usar split estratificado, deterministico e sem vazamento entre treino e teste.
- Avaliar Macro-F1 e matriz de confusao.
- Exigir Macro-F1 superior ao majority baseline.
- Exportar o classificador aprovado para ONNX e comparar suas classes com as do mesmo classificador scikit-learn no conjunto de teste.
- Medir o caminho completo de inferencia, do texto ate a classe, com warm-up, mediana e p95, comparando o modelo original e o otimizado sob as mesmas condicoes.
- Medir tambem o tempo de resposta HTTP da API em Docker para o modelo original e o otimizado.
- Se ONNX nao trouxer ganho, selecionar `ccp_alpha` sem consultar o conjunto de teste, reavaliar a qualidade do candidato podado, comparar esse candidato com sua propria exportacao ONNX e repetir o benchmark.

### 3.3 MLflow

- Executar um servidor MLflow local com backend SQLite e artefatos em volume persistente.
- Registrar parametros, metricas, hash DVC, benchmark e artefatos de avaliacao.
- Registrar o modelo como `triage-urgency-classifier`.
- Empacotar um MLflow PyFunc que carregue o `TfidfVectorizer`, o classificador ONNX e o mapeamento das classes.
- Usar o alias `champion` para identificar a versao servida.
- Promover uma versao para `champion` apenas depois dos gates pre-promocao de dados, qualidade, equivalencia, benchmark e integridade do bundle.
- Aplicar uma nova versao por alteracao do alias e reinicio da API, sem hot-swap.

### 3.4 API

- Implementar `POST /predict` para receber texto e retornar classe e versao do modelo.
- Implementar `GET /health` para informar disponibilidade e versao carregada.
- Implementar `GET /metrics` para expor metricas Prometheus.
- Carregar `models:/triage-urgency-classifier@champion` uma unica vez no startup.
- Falhar explicitamente no startup quando o alias ou o artefato nao estiver disponivel.
- Validar entrada vazia e tamanho maximo sem incluir o texto recebido nas mensagens de erro.

### 3.5 Orquestracao e infraestrutura

- Criar uma DAG manual do Airflow com as etapas `validate_data`, `train_and_evaluate`, `optimize_and_benchmark` e `register_and_promote`.
- Executar Airflow standalone localmente, fora do Docker Compose do MVP.
- Criar Dockerfile funcional para a API.
- Criar Docker Compose com API, MLflow, Prometheus e Grafana.
- Provisionar um dashboard Grafana com total de requisicoes, latencia p95 e taxa de erro.

### 3.6 CI/CD e entrega

- Executar Ruff format/check, mypy e pytest em push e pull request.
- Construir a imagem Docker no GitHub Actions.
- Documentar arquitetura, execucao, resultados, limitacoes e atribuicao do dataset no README.
- Documentar ao final a escolha teorica de nuvem para uma API de inferencia em tempo real.
- Gravar um video de ate cinco minutos usando o metodo STAR.

## 4. Fora do escopo

- Validade clinica, diagnostico ou recomendacao medica.
- Deploy real em provedor de nuvem.
- Kubernetes, alta disponibilidade e escalabilidade automatica.
- Autenticacao, autorizacao e gestao de usuarios.
- Hot-swap de modelos em runtime.
- Endpoint administrativo para promocao de modelo.
- Probabilidades ou scores de confianca na resposta do MVP.
- Monitoramento de drift ou qualidade dos dados de producao.
- Retreino automatico por agenda ou evento.
- Migracao de DVC e artefatos MLflow para S3.

## 5. Fluxo principal

1. A versao fixada do dataset e recuperada pelo DVC.
2. O gate valida origem, checksum, licenca, schema e qualidade minima.
3. `train_and_evaluate` treina o baseline e o modelo principal e registra a avaliacao.
4. `optimize_and_benchmark` converte o candidato, verifica equivalencia e compara desempenho; se necessario, avalia o candidato podado sem selecionar parametros no teste.
5. `register_and_promote` registra o bundle aprovado, confere seus hashes e associa o alias `champion`.
6. A API carrega o `champion` durante o startup e confirma a versao servida.
7. O Prometheus coleta metricas e o Grafana apresenta o dashboard.

### 5.1 Bootstrap local

Em uma instalacao vazia, a ordem de inicializacao sera:

1. Subir somente o servico MLflow com seu backend e artifact store persistentes.
2. Executar a DAG standalone apontando para o tracking server e criar a primeira versao `champion`.
3. Subir API, Prometheus e Grafana depois que o alias puder ser resolvido.

O MLflow funcionara como proxy de artefatos. Airflow e API acessarao modelos pelo tracking server HTTP, sem depender de um caminho de filesystem exclusivo de outro container. Depois do primeiro bootstrap, a stack completa podera ser reiniciada normalmente porque SQLite e artefatos permanecerao nos volumes.

## 6. Gates de aceite

### Gate de dados

- Fonte e versao identificadas.
- Arquivo e checksum registrados.
- Licenca documentada de forma conservadora.
- Pelo menos 2.000 amostras validas.
- Colunas de texto em ingles e urgencia presentes.
- Exatamente os targets publicados `low`, `medium` e `high`.
- Ausencia de nulos nos campos usados pelo modelo.
- Duplicatas medidas e tratadas antes do split.

Os nomes de colunas e targets acima descrevem o KurMed-Triage. Se o fallback for necessario, plano, ADR e card deverao ser revisados antes da modelagem para registrar o schema e o mapeamento publicados pela nova fonte; os demais controles do gate permanecem obrigatorios.

### Gate de qualidade

- Split estratificado e reproduzivel.
- Majority baseline registrado.
- Macro-F1 do modelo principal superior ao baseline.
- Matriz de confusao e distribuicao das classes documentadas.

### Gate de otimizacao

- Para cada candidato, classes da exportacao ONNX equivalentes as classes do mesmo classificador scikit-learn no conjunto de teste.
- Se houver pruning, `ccp_alpha` selecionado em treino/validacao e o candidato novamente aprovado no gate de qualidade sobre o teste intocado.
- Benchmark in-process com 20 execucoes de warm-up e 500 predicoes individuais medidas sobre uma sequencia deterministica de textos.
- Benchmark HTTP em Docker com 20 requisicoes de warm-up e 200 requisicoes sequenciais medidas, um worker e a mesma maquina, amostra e configuracao para original e otimizado.
- Mediana e p95 registrados; ganho aceito quando a mediana melhora pelo menos 5% e o p95 nao regride mais de 5%.

### Gate de integridade e promocao

- O bundle PyFunc contem o TF-IDF e o classificador ONNX aprovados no mesmo run avaliado.
- Hashes dos artefatos, run ID e hash DVC registrados como tags da versao.
- Assinatura do PyFunc aceita um DataFrame com uma coluna string `text`, uma ou mais linhas, e retorna um DataFrame com a coluna string `classification` na mesma ordem.
- Matriz TF-IDF convertida para tensor denso `float32` no formato `[batch, features]` esperado pelo ONNX.
- Nomes dos tensores e ordem das classes persistidos e cobertos por teste de lote multiclasse.
- Somente depois dessas verificacoes a versao recebe o alias `champion`.

### Gate de servico

- API carrega o alias `champion` no startup.
- Contratos de `/predict`, `/health` e `/metrics` cobertos por testes.
- Imagem Docker funcional.
- Stack local sobe com API, MLflow, Prometheus e Grafana.
- Dashboard possui os tres paineis obrigatorios.
- A versao retornada pela API corresponde a versao do bundle carregado pelo alias e aos hashes registrados no MLflow.
- Um texto-canario submetido nao aparece em respostas de erro, logs da aplicacao ou servidor, metricas, tags, parametros ou metadados do MLflow.

### Gate de entrega

- CI verde para lint, tipos, testes e build Docker.
- DAG importada sem erros e executada manualmente de ponta a ponta, com dependencias respeitadas, metricas registradas e uma versao verificavel no Model Registry.
- README permite reproduzir a demonstracao.
- Comparativo de latencia e decisao de nuvem documentados.
- Roteiro STAR e video final concluidos.

## 7. Arquitetura proposta

```text
KurMed-Triage v1
       |
      DVC
       |
validate_data -> train_and_evaluate -> optimize_and_benchmark
                                             |
                                  register_and_promote
                                             |
                                        MLflow Server
                               SQLite + proxied artifacts
                                             |
                         triage-urgency-classifier@champion
                                             |
                                          FastAPI
                                             |
                                         Prometheus
                                             |
                                           Grafana
```

## 8. Backlog e cronograma

| Ordem | Entrega | Estimativa | Inicio | Prazo | Prioridade |
|---:|---|---:|---|---|---|
| 1 | Estruturar projeto, documentacao inicial e regras | 1h30 | 12/08 | 14/08 | Alta |
| 2 | Validar KurMed-Triage e configurar DVC | 2h30 | 15/08 | 18/08 | Maxima |
| 3 | Treinar e avaliar baseline NLP | 3h | 18/08 | 21/08 | Maxima |
| 4 | Converter para ONNX e medir latencia | 3h | 22/08 | 25/08 | Maxima |
| 5 | Integrar MLflow Tracking e Model Registry | 2h30 | 25/08 | 27/08 | Alta |
| 6 | Implementar FastAPI e testes | 3h | 28/08 | 31/08 | Maxima |
| 7 | Criar Dockerfile e validar integracao local | 1h30 | 01/09 | 02/09 | Alta |
| 8 | Implementar DAG Airflow | 2h30 | 02/09 | 04/09 | Alta |
| 9 | Configurar Prometheus e Grafana | 3h | 05/09 | 09/09 | Maxima |
| 10 | Configurar CI no GitHub Actions | 1h30 | 09/09 | 10/09 | Alta |
| 11 | Finalizar README, decisao cloud e video | 3h | 11/09 | 15/09 | Maxima |

Estimativa total: 27 horas.

A distribuicao prevista e de ate 5h30 nas quatro primeiras semanas e 5h na semana final. As estimativas assumem o KurMed aprovado e ONNX suficiente. Acionar fallback de dataset ou pruning exige reestimar os cards afetados e decidir explicitamente o que sera movido para pos-MVP; essas contingencias nao estao escondidas nas 27 horas.

### Pos-MVP

| Entrega | Prazo alvo | Prioridade |
|---|---|---|
| Adicionar probabilidades calibradas | 30/09/2026 | Baixa |
| Migrar DVC e artefatos MLflow para S3 | 15/10/2026 | Baixa |

## 9. Riscos e mitigacoes

| Risco | Impacto | Mitigacao |
|---|---|---|
| Divergencia de licenca entre Kaggle e Hugging Face | Uso inadequado do dataset | Adotar provisoriamente CC BY-SA 4.0, manter atribuicao e verificar o arquivo de licenca antes da aprovacao |
| Dataset sintetico com baixa representatividade clinica | Metricas pouco generalizaveis | Declarar a limitacao e restringir o produto a demonstracao educacional |
| Duplicatas causarem vazamento | Metricas infladas | Detectar duplicatas antes do split e manter grupos relacionados no mesmo conjunto |
| Random Forest nao ganhar latencia com ONNX | Criterio de otimizacao nao atendido | Aplicar pruning com `ccp_alpha`, reconverter e repetir o benchmark |
| MLflow ampliar o escopo | Atraso no MVP | Limitar a SQLite, artefatos locais, um modelo, um alias e restart manual |
| Dependencia do Google Drive | Falha na reproducao | Registrar origem, versao e checksum do dataset e dos artefatos |
| Pouca margem no cronograma | Entrega incompleta | Implementar os cards na ordem da rubrica e mover apenas extras para pos-MVP se necessario |

## 10. Definicao de pronto do MVP

O MVP esta pronto quando todos os gates estiverem aprovados, os requisitos obrigatorios da rubrica puderem ser demonstrados localmente, o repositorio contiver instrucoes reproduziveis e o video STAR apresentar o fluxo completo em ate cinco minutos.

## 11. Decisoes relacionadas

- [ADR 0001 - Dominio e dataset](adr/0001-dominio-e-dataset.md)
- [ADR 0002 - Arquitetura do MVP](adr/0002-arquitetura-do-mvp.md)
- [Glossario](glossario.md)
