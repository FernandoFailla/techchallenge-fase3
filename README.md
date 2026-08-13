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
```
