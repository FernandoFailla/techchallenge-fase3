# Decisão Teórica De Cloud

- Status: proposta teórica; nenhum recurso de cloud foi provisionado.
- Decisão: para uma futura API de inferência em tempo real, avaliar Google Cloud
  Run como destino inicial do contêiner FastAPI.

## Contexto E Justificativa

O caso de uso exige resposta por requisição, portanto batch não atende ao
contrato da API. A aplicação atual já é um contêiner stateless, expõe HTTP e
carrega uma versão imutável do modelo no startup. Um serviço gerenciado de
containers reduz a operação inicial em comparação com manter VMs ou Kubernetes,
sem mudar o contrato FastAPI.

Cloud Run é a opção inicial proposta por essa compatibilidade. Não foram medidos
custo, disponibilidade, latência, escalabilidade ou cold start em cloud; a
decisão não é uma recomendação de produção nem uma implementação realizada.

## Condições Para Implementar

- Publicar imagem imutável em registry e configurar URI/versionamento do modelo
  por ambiente, sem depender do volume Docker local.
- Substituir SQLite, volumes locais e Google Drive por stores gerenciados com
  backup, controle de acesso e retenção definidos.
- Usar gestão de segredos e identidade de serviço; nunca incluir credenciais na
  imagem ou em variáveis versionadas.
- Definir autenticação da API, limite de tamanho, rede, logs sem texto clínico,
  orçamento, SLO e estratégia de rollback antes de exposição externa.
- Medir p50, p95, cold start, concorrência, erros e custo sob carga representativa
  antes de comparar provedores ou aprovar o deploy.

## Alternativas

- Serviço de batch: rejeitado para a classificação interativa; pode servir para
  treinamento ou reprocessamento futuro.
- VM gerenciada: possível, mas adiciona operação de sistema, patching e
  escalabilidade ao MVP.
- Kubernetes: adiado; replicas, autoscaling e rollout aumentam a complexidade
  sem evidência de necessidade neste protótipo.

## Consequência

A arquitetura local permanece a referência reproduzível da entrega. Uma migração
para cloud requer ADR atualizado, avaliação de segurança/custos e nova bateria
de medições; ela não pode reutilizar os resultados locais como prova de
desempenho em cloud.
