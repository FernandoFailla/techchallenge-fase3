# ADR 0001: Dominio e dataset

- Status: Aceita com gate de validacao pendente
- Data: 12/08/2026
- Decisores: Fernando Failla

## Contexto

O Tech Challenge exige um classificador textual de urgencia medica com pelo menos 2.000 amostras. O projeto deve ser concluido individualmente em poucas semanas e seu objetivo principal e demonstrar deploy, CI/CD, orquestracao, monitoramento e otimizacao de latencia, nao desenvolver um sistema clinico.

Durante o grilling foi estabelecido que o target nao sera criado pelo projeto. O corpus deve ter sido publicado e rotulado por terceiros. Traduzir textos, agregar classes ou anotar urgencia localmente produziria um problema novo e fragilizaria a rastreabilidade da demonstracao.

O KurMed-Triage v1 foi encontrado em Kaggle e Hugging Face. As duas fontes descrevem 2.000 casos sinteticos, uma coluna de texto em ingles e tres niveis de urgencia. Seus metadados de licenca divergem: o Kaggle cataloga o dataset como CC BY-SA 4.0, enquanto o Hugging Face e a descricao do Kaggle informam CC BY 4.0.

## Decisao

O produto classificara uma queixa textual em ingles em uma das classes de negocio:

- `normal`
- `atencao`
- `urgente`

O KurMed-Triage v1 sera a fonte principal, condicionado a um gate de validacao. Sera usada a coluna `patient_text_en` como entrada e o target publicado `urgency`, com o seguinte mapeamento de apresentacao:

| Target publicado | Classe do produto |
|---|---|
| `low` | `normal` |
| `medium` | `atencao` |
| `high` | `urgente` |

Esse mapeamento apenas traduz nomes de dominio para portugues e preserva a cardinalidade e o significado dos targets publicados. O projeto nao inferira nem alterara a urgencia de nenhum registro.

O dataset sera aceito somente se a versao analisada:

- puder ser obtida de uma fonte publica identificada;
- contiver pelo menos 2.000 registros;
- possuir `patient_text_en` e `urgency` preenchidos;
- possuir exatamente `low`, `medium` e `high` como targets usados;
- tiver origem, versao e checksum registrados;
- tiver nulos, distribuicao de classes, IDs duplicados e textos duplicados medidos;
- permitir um split sem vazamento conhecido;
- tiver sua licenca e atribuicao documentadas.

Enquanto a divergencia nao for resolvida por um arquivo de licenca inequivoco, o projeto obedecera a interpretacao mais restritiva, CC BY-SA 4.0, e atribuira o trabalho a Alan Jafari. A documentacao registrara a divergencia em vez de apresentar uma licenca como fato incontroverso.

Dados e artefatos finais serao rastreados com DVC. O primeiro remote sera um Google Drive compartilhado. Fonte original, versao e checksum permanecerao documentados para permitir recuperacao independente do remote.

## Alternativas consideradas

### Criar ou anotar um dataset proprio

Rejeitada. Exigiria conhecimento clinico, introduziria targets sem validacao independente e desviaria o foco da rubrica.

### Traduzir um corpus de outro idioma

Rejeitada. A traducao seria uma transformacao autoral adicional e poderia mudar pistas relevantes de urgencia.

### Agregar cinco niveis de triagem em tres classes

Rejeitada. A agregacao criaria uma politica de rotulagem local, contrariando a restricao de usar targets publicados sem alteracao.

### MIMIC-IV-Note ou MIMIC-CXR

Rejeitada. Esses conjuntos nao fornecem diretamente o target de urgencia requerido.

### MIMIC-IV-ED

Rejeitada para o MVP. Exige credenciamento e o uso considerado demandaria agregar cinco classes de triagem em tres.

### Medical Abstracts TC Corpus

Nao escolhido como fonte principal porque suas classes representam especialidades ou categorias medicas, e nao exatamente os tres niveis de urgencia desejados.

### Fallback com outro corpus de tres classes

Aceita apenas como contingencia. Se o KurMed falhar no gate, a substituicao deve ser publicada, acessivel e previamente rotulada com exatamente tres classes semanticamente equivalentes a urgencia. Antes da modelagem, a troca exigira atualizar este ADR, o plano e o card de dados com fonte, schema, targets e mapeamento efetivamente usados.

## Consequencias

### Positivas

- O problema de modelagem permanece simples e alinhado ao prazo.
- Os targets possuem autoria externa e sao rastreaveis.
- As classes da API correspondem diretamente aos tres niveis publicados.
- O uso de texto em ingles reduz complexidade de tokenizacao e demonstracao.

### Negativas

- O dataset e sintetico e muito recente, com pouca evidencia externa de qualidade.
- A divergencia de licenca exige postura conservadora.
- O contexto cultural do corpus limita a generalizacao.
- Um corpus de 2.000 registros pode produzir metricas instaveis ou sensiveis a duplicatas.

## Limitacoes eticas e de seguranca

- O sistema e exclusivamente educacional.
- Nenhuma metrica obtida sera apresentada como evidencia de seguranca clinica.
- A resposta da API nao substitui avaliacao profissional.
- Os textos recebidos nao serao registrados em logs, Prometheus, MLflow ou mensagens de erro.
- A documentacao final deve destacar vieses, origem sintetica e limites de generalizacao.

## Evidencias conhecidas

- Kaggle: `https://www.kaggle.com/datasets/alanjafari/kurmed-triage`
- Hugging Face: `https://huggingface.co/datasets/alanjafari/KurMed-Triage`
- Arquivo observado no Hugging Face: `synthetic_v1.jsonl`
- Autor informado: Alan Jafari
- Versao informada: v1

## Criterio para encerrar a pendencia

O status deixa de conter a ressalva de gate pendente quando o card de validacao registrar o arquivo efetivamente usado, seu checksum, schema, contagens, distribuicao, duplicatas e conclusao sobre a licenca aplicavel.
