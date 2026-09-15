---
marp: true
theme: default
paginate: true
title: Tech Challenge - Triagem Textual Com MLOps
---

# Triagem Textual Com MLOps

## Modelo rapido, rastreavel e monitorado

<!-- PRESENTATION_METADATA -->

<!-- Notas: Prototipo educacional com dados sinteticos. Sem uso clinico. -->

---

# S | Situation

## Classificar urgencia em tempo real

```text
texto -> low | medium | high
```

- 2.000 registros sinteticos
- resposta por requisicao, nao batch
- rapido nao basta: precisa ser auditavel

<!-- Notas: O problema nao e apenas prever. A resposta precisa ser rapida e auditavel. -->

---

# T | Task

## Entregar o ciclo MLOps completo

| Construir | Comprovar |
|---|---|
| API em Docker | CI verde |
| Pipeline Airflow | treino reproduzivel |
| ONNX Runtime | mesma classe, menor latencia |
| Prometheus + Grafana | trafego, P95 e erros |

<!-- Notas: Resuma a fase em uma frase: automatizar, otimizar e observar. -->

---

# A | Pipeline de treino

```text
validate_modeling_base
  -> train_and_evaluate
  -> optimize_and_benchmark
  -> register_and_promote
```

- split deterministico, seed 42
- promocao so depois dos gates
- real-time por requisicao; Cloud Run como proposta teorica

<!-- PRESENTATION_QUALITY -->

**Demo:** DAG `training_pipeline` concluida no Airflow

<!-- Notas: Mostre as quatro tarefas verdes. Os nomes no slide sao os mesmos da tela do Airflow. -->

---

# A | Otimizacao ONNX

Mesmo texto, mesmo TF-IDF, mesmo Random Forest.

<!-- PRESENTATION_ONNX_RESULTS -->

**Demo:** runs e alias `champion` no MLflow

<!-- Notas: Primeiro paridade; depois velocidade. O benchmark mede o caminho texto ate classe. -->

---

# A | API Observavel

```text
FastAPI -> Prometheus -> Grafana
```

| Trafego | Desempenho | Erros |
|---|---|---|
| total | P95 | taxa 5xx |

**Demo:** dashboard `Observabilidade da API` no Grafana

<!-- Notas: Total maior que zero. Assim, 0% de 5xx confirma ausencia de erros internos. -->

---

# R | Result

<!-- PRESENTATION_RESULT -->

**Fechando o ciclo:** um texto entra, uma classe sai - do dado versionado ao deploy, otimizado sem mudar a resposta e observado em operacao.

**Limites:** dados sinteticos, sem validacao clinica, sem deploy cloud real.

<!-- Notas: O HTTP mede o champion completo. O ganho causal do ONNX vem do slide anterior. Feche retomando o problema do inicio, sem alegar prontidao clinica. -->
