# Roteiro E Checklist STAR

Meta: vídeo de até cinco minutos. Mostre somente dados sintéticos e agregados;
não exiba `.env`, tokens, textos clínicos, IDs ou artefatos que possam reter
vocabulário treinado.

## Roteiro

| Tempo | STAR | Fala e demonstração |
|---|---|---|
| 0:00-0:45 | Situation | Apresente o protótipo educacional de classificação de urgência textual e declare que não tem uso clínico. Mostre o diagrama do README. |
| 0:45-1:15 | Task | Explique os requisitos: dados rastreáveis, treino orquestrado, API em contêiner, CI, métricas e otimização de latência. |
| 1:15-2:20 | Action | Mostre os ponteiros DVC, MLflow e a DAG `training_pipeline`; explique a sequência validar, treinar, converter para ONNX, verificar e promover `champion`. |
| 2:20-3:20 | Action | Mostre `make observability`, `/health`, Prometheus e os três painéis Grafana. Explique que a API carrega uma versão no startup e que labels não incluem texto. |
| 3:20-4:15 | Result | Execute ou mostre `make api-benchmark`. Informe o snapshot do README como medição local e host-específica: Macro-F1 de teste `0.758`, paridade ONNX `1.000`, e HTTP P95 `2.998 ms`. |
| 4:15-5:00 | Result | Mostre CI e resuma limites: dataset sintético, licença pendente de confirmação, sem uso clínico e cloud apenas teórica. Indique os próximos passos de validação. |

## Checklist Antes De Gravar

- [ ] `make check` terminou sem falhas.
- [ ] `make docker-config` terminou sem falhas.
- [ ] A DAG manual concluiu e existe `triage-urgency-classifier@champion`.
- [ ] `/health` responde sem enviar texto clínico.
- [ ] API, Prometheus e Grafana estão saudáveis; dashboard contém total, p95 e 5xx.
- [ ] O snapshot de métricas mostra data, protocolo, versão do modelo e ressalva de máquina local.
- [ ] A tela não mostra `.env`, senha do Airflow, token Kaggle, OAuth, URLs privadas ou textos do dataset.
- [ ] A duração ensaiada é menor ou igual a cinco minutos.

## Após Gravar

- [ ] Rever o vídeo procurando segredos e dados sensíveis na tela ou no áudio.
- [ ] Confirmar que resultados são apresentados como medições locais, não como garantia clínica ou de produção.
- [ ] Publicar o link solicitado pela avaliação e registrar a versão/commit demonstrada.
