# Roteiro Final STAR

Meta: video de ate cinco minutos. Mostre somente dados sinteticos e agregados.
Nao exiba `.env`, tokens, senhas, IDs de registros, textos clinicos ou artefatos
que possam reter vocabulario treinado.

## Preparacao

Execute estes passos antes de iniciar a gravacao e deixe apenas as telas seguras
abertas:

```bash
make check
make docker-config
make mlflow
```

Em outro terminal, execute `make airflow`, dispare a DAG `training_pipeline` e
aguarde todas as tarefas concluirem. Depois, inicie a stack final:

```bash
make observability
curl --fail http://localhost:8000/health
make api-load-test
```

Aguarde uma coleta do Prometheus antes de abrir Grafana. Tenha tambem abertas as
paginas MLflow, Airflow, Grafana e o resultado verde do GitHub Actions do commit
que sera demonstrado.

## Roteiro De Cinco Minutos

| Tempo | STAR | Fala e demonstracao |
|---|---|---|
| 0:00-0:35 | Situation | "Este e um prototipo educacional de triagem textual. Ele usa dados sinteticos em ingles e nao tem uso clinico." Mostre o diagrama do README. |
| 0:35-1:05 | Task | "O objetivo foi demonstrar um ciclo MLOps reproduzivel: dados versionados, treino, registro de modelo, API, observabilidade, CI e otimizacao." Mostre os ponteiros DVC, sem abrir dados. |
| 1:05-2:00 | Action | Mostre a DAG `training_pipeline` concluida. Explique as quatro etapas: validar a base, treinar e avaliar, converter e comparar ONNX, registrar e promover o alias `champion`. |
| 2:00-2:50 | Action | Mostre MLflow: experimento, metricas agregadas, modelo `triage-urgency-classifier` e alias `champion`. Diga a versao exibida por `/health`; nao mostre textos nem artefatos de vocabulario. |
| 2:50-3:40 | Action | Mostre Grafana apos `make api-load-test`: total de requisicoes, P95 e taxa 5xx. Explique que 0% de 5xx significa ausencia de erros internos, nao falta de dados. |
| 3:40-4:25 | Result | Mostre os resultados da execucao atual no MLflow: Macro-F1 do teste, paridade ONNX e metricas de latencia. Declare o protocolo ONNX: 20 warm-ups, 500 predicoes individuais, media, P50 e P95. Declare que latencia depende da maquina. |
| 4:25-5:00 | Result | Mostre CI verde. Resuma limites: corpus sintetico, politica conservadora CC BY-SA 4.0 por divergencia de metadados, sem uso clinico, sem deploy cloud real e sem probabilidades calibradas. |

## Valores A Preencher Antes Da Gravacao

Copie somente valores agregados da run mais recente. Nao reutilize o snapshot
historico do README depois de executar o novo protocolo ONNX.

| Evidencia | Valor da run atual |
|---|---|
| Commit demonstrado | `[preencher]` |
| Versao `champion` | `[preencher]` |
| Macro-F1 no teste | `[preencher]` |
| Paridade ONNX no teste | `[preencher]` |
| ONNX P50 e P95 | `[preencher]` |
| API HTTP P95 | `[preencher]` |

## Checklist Antes De Gravar

- [ ] `make check` e `make docker-config` terminaram sem falhas.
- [ ] A DAG manual concluiu e existe `triage-urgency-classifier@champion`.
- [ ] O benchmark ONNX da run demonstrada usa 20 warm-ups e 500 predicoes.
- [ ] `/health` responde e a versao coincide com o alias no MLflow.
- [ ] Grafana mostra total, P95 e taxa 5xx; sem erros 5xx, a taxa mostra `0%`.
- [ ] O commit, a versao do modelo, data e protocolo das metricas foram anotados.
- [ ] A tela nao mostra `.env`, senha do Airflow, token Kaggle, OAuth, URLs privadas ou textos do dataset.
- [ ] O ensaio dura no maximo cinco minutos.

## Apos Gravar

- [ ] Rever video e audio procurando segredos e dados sensiveis.
- [ ] Declarar todas as metricas como observacoes locais, nunca como garantia clinica ou de producao.
- [ ] Publicar o link solicitado pela avaliacao e registrar o commit demonstrado.
