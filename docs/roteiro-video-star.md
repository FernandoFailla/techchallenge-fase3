# Roteiro Final Do Video STAR

Meta: demonstrar em ate cinco minutos todos os criterios de `techchallengefase3.md`
com evidencias atuais e verificadas. Mostre somente dados sinteticos e agregados.
Nao exiba `.env`, tokens, senhas, IDs de registros, textos do dataset, vetores
TF-IDF ou artefatos que retenham vocabulario treinado.

## Entregaveis Para Usar Na Gravacao

- Template versionado dos slides: `docs/apresentacao-slides.md`.
- Slides preenchidos com metricas locais: `presentation-slides.generated.md`.
- Relatorio de metricas: saida de `make presentation-report`.
- DAG: `training_pipeline` no Airflow.
- Modelo: `triage-urgency-classifier@champion` no MLflow.
- Dashboard: `Observabilidade da API` no Grafana.
- CI: ultima execucao do workflow `CI` no GitHub Actions.

## Preparacao Uma Unica Vez

```bash
make setup
```

Se o caminho DVC for usado, `make setup` recupera a base aprovada. Se o caminho
Kaggle for usado, ele recupera apenas o CSV publico; nesse caso, inicie o Airflow
e execute `prepare_modeling_base` antes do pipeline de treino.

## Atualizar Todas As Evidencias

Use tres terminais. Os servicos dos terminais 1 e 2 devem permanecer ativos.

### Terminal 1: MLflow

```bash
make mlflow
```

Abra `http://localhost:5000` e confirme que o servidor responde.

### Terminal 2: Airflow

```bash
make airflow
```

Em outro terminal, obtenha a senha com `make airflow-password`, mas nunca a
mostre no video. Abra `http://localhost:8080`, execute `training_pipeline` e
aguarde as quatro tarefas terminarem:

```text
validate_modeling_base
  -> train_and_evaluate
  -> optimize_and_benchmark
  -> register_and_promote
```

Essa execucao atualiza qualidade, benchmark ONNX, paridade, bundle registrado e
alias `champion`. Nao use o snapshot historico do README como resultado atual.

### Terminal 3: stack final e evidencias

```bash
make presentation-evidence
```

O comando inicia a stack final em modo detached, aguarda os servicos ficarem
prontos e entao coleta as evidencias. Assim, o terminal permanece disponivel.

`make presentation-evidence` executa, nesta ordem:

1. Ruff, mypy strict, pytest e verificacao de formato.
2. Validacao do Docker Compose.
3. Benchmark HTTP com 20 warm-ups e 200 requisicoes medidas.
4. Carga sintetica concorrente para preencher os paineis do Grafana.
5. Relatorio consolidado das runs mais recentes do MLflow.
6. Geracao de `presentation-slides.generated.md` com os valores atuais.

Apresente o arquivo gerado, nao o template em `docs/`. Os pontos de injecao de
metricas sao comentarios invisiveis e o comando os substitui pelos valores.

Aguarde pelo menos uma coleta de 15 segundos do Prometheus. Se as metricas ja
estiverem atualizadas e voce quiser apenas consultar ou regerar os slides, use:

```bash
make presentation-report
make presentation-slides
```

O relatorio deve mostrar `OK` para:

- codigo demonstrado sem alteracoes locais relevantes;
- protocolo ONNX de 20 warm-ups e 500 medicoes;
- paridade ONNX de 100%;
- ganho medio ONNX positivo;
- versao HTTP igual ao alias `champion`;
- mesmo hash DVC nas runs de qualidade, ONNX, paridade e registro.
- configuracao do modelo avaliado igual a configuracao do `champion`.

Se aparecer `PENDENTE`, corrija ou reexecute a etapa indicada. Nao esconda a
divergencia e nao transforme ausencia de metrica em zero.

## Telas Seguras A Deixar Abertas

Na ordem em que aparecem na gravacao:

1. `presentation-slides.generated.md` em modo de apresentacao.
2. Airflow com a ultima run de `training_pipeline` concluida.
3. MLflow mostrando as runs do `kan-10-onnx-benchmark` e o alias `champion`.
4. Grafana no dashboard `Observabilidade da API`, intervalo `Last 15 minutes`.
5. GitHub Actions com o workflow `CI` verde no commit demonstrado.
6. Terminal com a saida de `make presentation-report`, se houver tempo.

Feche `.env`, senha do Airflow, console OAuth, arquivos de dados e artifacts do
modelo antes de iniciar a captura.

## Mapa Dos Requisitos Oficiais

| Requisito de `techchallengefase3.md` | Evidencia no projeto | Demonstracao rapida |
|---|---|---|
| API FastAPI recebe texto e retorna classe | `src/techchallenge/api.py` e container | `/health` e uma chamada sintetica a `/predict` |
| Dockerfile funcional | `Dockerfile.api` | CI verde na etapa `Build API image` |
| Decisao cloud e real-time versus batch | `docs/decisao-cloud.md` | slide 4 do pipeline |
| CI/CD com pelo menos duas automacoes | `.github/workflows/ci.yml` | Ruff, mypy, pytest e build no Actions |
| DAG Airflow de treino | `dags/training_pipeline.py` | quatro tarefas verdes no Airflow |
| Modelo de texto | TF-IDF + Random Forest | Macro-F1 e acuracia no relatorio |
| Otimizacao de latencia | Random Forest no ONNX Runtime | tabela sklearn versus ONNX |
| Equivalencia da otimizacao | gate de paridade no teste | paridade `1.000` antes do speedup |
| Stack API + Prometheus + Grafana | `compose.mlflow.yml` | servicos ativos e dashboard |
| Total de requisicoes | painel Grafana 1 | carga gerada por `make api-load-test` |
| Latencia de resposta | painel Grafana 2 e benchmark HTTP | P95 no Grafana e no relatorio |
| Taxa de erro | painel Grafana 3 | com total maior que zero, `0%` significa nenhuma resposta 5xx |
| Historico semantico | commits Conventional Commits | commit exibido no relatorio e GitHub |
| Resultado comparativo | MLflow + `make presentation-report` | media, P50, P95, reducao e speedup |
| Video STAR | este roteiro | Situation, Task, Action e Result em 5 min |

## Roteiro Cronometrado De Cinco Minutos

Cada slide de acao termina com a demonstracao ao vivo que o segue. Os nomes no
slide 4 sao os mesmos nomes de tarefas exibidos no Airflow; use isso como ancora
visual ao trocar de tela.

### 0:00-0:30 - Situation

**Tela:** slides 1 e 2.

**Fala:**

> Este projeto responde ao cenario de um hospital que precisa classificar a
> urgencia de textos com baixa latencia. Construi um prototipo educacional de
> inferencia em tempo real usando 2.000 registros sinteticos em ingles. Ele nao
> e dispositivo medico e nao deve orientar diagnostico ou atendimento real.

### 0:30-0:50 - Task

**Tela:** slide 3.

**Fala:**

> A tarefa nao era apenas treinar um classificador. A entrega precisava integrar
> API REST em Docker, CI/CD, orquestracao Airflow, monitoramento Prometheus e
> Grafana, alem de demonstrar uma otimizacao real de latencia com rastreabilidade.

### 0:50-1:35 - Action: pipeline de treino

**Tela:** slide 4; ao final, troque para o Airflow com a run concluida.

**Fala:**

> O pipeline e orquestrado por quatro tarefas: validar a base, treinar e avaliar,
> converter e comparar ONNX, e so entao registrar e promover o alias champion.
> Como a classificacao exige resposta por requisicao, escolhi real-time; para uma
> futura nuvem, a proposta teorica e Cloud Run. Aqui esta a DAG concluida, do
> inicio ao fim, na mesma execucao.

### 1:35-2:25 - Action: otimizacao ONNX

**Tela:** slide 5; ao final, troque para o MLflow com as runs e o alias.

**Fala:**

> O TF-IDF permanece em Python e o Random Forest e executado pelo ONNX Runtime.
> Primeiro verifico paridade: as mesmas classes nas mesmas entradas do teste. So
> depois comparo latencia, com 20 warm-ups e 500 predicoes individuais. A tabela
> mostra media, P50 e P95 antes e depois. No MLflow, estas sao as runs do
> benchmark e o bundle registrado como champion.

### 2:25-3:05 - Action: observabilidade

**Tela:** slide 6; ao final, troque para o Grafana preenchido.

**Fala:**

> A API instrumentada expoe contagem e histograma de duracao. O Prometheus coleta
> a cada 15 segundos e o Grafana e provisionado como codigo. Os tres paineis
> mostram total de requisicoes, latencia P95 e taxa de erro 5xx. A carga foi
> gerada com dados sinteticos. Como o total e maior que zero, a taxa de zero por
> cento confirma que nao houve respostas 5xx nessa carga.

### 3:05-4:15 - Result: numeros

**Tela:** slide 7; se houver tempo, mostre o CI verde por poucos segundos.

**Fala:**

> Fechando o ciclo: o champion servido, o Macro-F1 do teste, o speedup do ONNX e
> a latencia HTTP da API otimizada. O relatorio confirma que a versao servida e
> o champion, que os hashes DVC sao iguais e que a configuracao avaliada
> corresponde ao modelo promovido. O benchmark HTTP mede o sistema completo; a
> atribuicao do ganho ao ONNX vem da comparacao equivalente do slide anterior.

Se o speedup for menor ou igual a `1.0x`, diga que o ganho esperado nao foi
demonstrado e nao apresente o gate como aprovado.

### 4:15-5:00 - Result: fechamento

**Tela:** mantenha o slide 7.

**Fala:**

> O resultado e um MVP reproduzivel: um texto entra, uma classe sai, do dado
> versionado ao deploy, otimizado sem mudar a resposta e observado em operacao.
> As principais licoes foram exigir equivalencia antes de otimizar e medir P50 e
> P95, nao apenas a media. Os limites sao corpus sintetico, sem probabilidades
> calibradas, sem deploy cloud real e, principalmente, sem qualquer garantia de
> uso clinico.

## Valores Que Devem Estar Preenchidos

Todos aparecem em `make presentation-report` e nos slides gerados.

| Evidencia | Fonte atual |
|---|---|
| Commit demonstrado | Git `HEAD` |
| Versao `champion` | alias no MLflow Model Registry |
| Acuracia e Macro-F1 | run final de `kan-11-baseline-nlp` |
| Protocolo ONNX | run final de `kan-10-onnx-benchmark` |
| Media, P50 e P95 sklearn/ONNX | run final de `kan-10-onnx-benchmark` |
| Reducao e speedup | calculados pelo relatorio |
| Paridade e correspondencias | run de paridade `kan-10-onnx-benchmark` |
| HTTP media, P50 e P95 | ultima run `kan-24-http-benchmark` |
| Total e taxa 5xx | dashboard Grafana apos a carga |
| CI lint/test/build | GitHub Actions do commit demonstrado |

## Checklist Antes De Gravar

- [ ] `make presentation-evidence` terminou sem falhas.
- [ ] Todos os checks do relatorio aparecem como `OK`.
- [ ] A DAG demonstrada terminou e existe `triage-urgency-classifier@champion`.
- [ ] Os slides gerados nao possuem `nao registrado`.
- [ ] `/health` informa a mesma versao do alias no MLflow.
- [ ] Grafana mostra total, P95 e taxa 5xx no intervalo correto.
- [ ] GitHub Actions esta verde para o commit exibido no relatorio.
- [ ] A tela nao mostra segredos, textos do dataset ou dados pessoais.
- [ ] O ensaio completo dura no maximo cinco minutos.

## Apos Gravar

- [ ] Rever video e audio procurando segredos ou dados sensiveis.
- [ ] Declarar metricas como observacoes locais, nunca garantias de producao.
- [ ] Publicar o link do video no canal solicitado pela avaliacao.
- [ ] Registrar o commit, a versao champion e a data demonstrados.
