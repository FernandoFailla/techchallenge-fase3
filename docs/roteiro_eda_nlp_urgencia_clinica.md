# EDA para classificação de textos clínicos em três níveis de urgência

## 1. Conclusão executiva

Para este problema, a EDA mais importante não é a contagem de palavras. Antes de estudar vocabulário ou embeddings, é necessário demonstrar cinco coisas:

1. **O alvo clínico está bem definido:** urgência na chegada, urgência de comunicação de um laudo, risco de deterioração ou outro desfecho não são a mesma tarefa.
2. **Os três rótulos são ordenados e confiáveis:** `baixa < média < alta`; um erro `alta → baixa` deve ser tratado como mais grave que `alta → média`.
3. **O texto estava disponível no momento da decisão:** notas, diagnósticos, condutas ou desfechos registrados depois da triagem produzem vazamento temporal.
4. **Paciente, duplicatas e modelos de texto não atravessam os conjuntos:** a separação deve ser por paciente e, preferencialmente, também temporal.
5. **A avaliação reflete segurança clínica:** acurácia e ROC-AUC isoladas não bastam; é preciso medir recall da classe alta, subtriagem grave, macro-F1, erro ordinal e calibração.

Neste projeto, a estratégia principal definida é **TF-IDF seguido de modelos de árvore**. Por isso, a EDA também deve caracterizar a matriz esparsa, testar a estabilidade do vocabulário e comparar duas rotas: `TF-IDF esparso → árvore` e `TF-IDF → TruncatedSVD → árvore`. Um modelo linear será mantido apenas como controle de sanidade e referência de desempenho.

Uma revisão de 60 estudos de triagem em pronto atendimento encontrou que NLP agregado a dados estruturados tende a melhorar a classificação e que engenharia de atributos e correção de desbalanceamento são relevantes. Porém, os estudos apresentaram risco de viés importante. As métricas mais frequentes foram ROC-AUC (78,3%), sensibilidade (71,6%), precisão/PPV (58,3%), especificidade (56,6%), acurácia (41,6%), F1 e NPV (35%) e AUPRC (13,3%). O fato de AUPRC ter sido pouco usada não significa que seja dispensável: para uma classe urgente rara, ela costuma ser mais informativa que ROC-AUC.

## 2. Primeiro: definir exatamente o problema clínico

Preencha, antes de abrir o notebook:

| Item | Definição que precisa ser registrada |
|---|---|
| Unidade de previsão | Uma nota, um atendimento, um laudo ou um paciente? |
| Momento zero | Em que instante o sistema faria a previsão? |
| Texto permitido | Somente queixa/nota de triagem? Laudo final? Histórico anterior? |
| Alvo | Urgência assistencial do paciente ou urgência de comunicação do achado? |
| Escala original | ESI, Manchester, classificação institucional, revisão de especialistas etc. |
| Três classes | Regra clínica usada para obter baixa, média e alta |
| Usuário do modelo | Enfermeiro, médico, radiologista, regulação ou auditoria? |
| Ação consequente | Qual fluxo muda quando a previsão é alta? |

### Duas tarefas que não devem ser misturadas

- **Prontuário/nota de triagem:** classifica a urgência do paciente em um momento inicial do atendimento. Textos posteriores ao momento zero não podem ser preditores.
- **Laudo radiológico:** classifica a urgência de comunicação de um achado. Nesse caso, as seções `achados` e `impressão/conclusão` podem ser válidas, mas o alvo precisa representar a urgência do achado, e não simplesmente o estado geral do paciente.

A literatura de laudos é especialmente próxima do problema ordinal. Tiwari et al. trabalharam com quatro níveis — Normal, Yellow, Orange e Red — usando Bag-of-Words/TF-IDF e vários classificadores. Isso mostra a viabilidade do alvo por gravidade, mas não autoriza juntar automaticamente níveis para formar três classes; o mapeamento deve ser validado clinicamente.

## 3. EDA recomendada, em ordem de prioridade

### 3.1 Auditoria da coorte e da proveniência

Calcular e documentar:

- número de pacientes únicos, atendimentos, documentos e documentos por paciente;
- período de coleta, instituições, setores, especialidades, autores e tipos de nota;
- unidade observacional real e chaves de ligação;
- distribuição temporal do volume;
- quantidade de documentos por paciente e por atendimento;
- critérios de inclusão/exclusão e um fluxograma da coorte;
- origem de cada campo e instante em que ficou disponível;
- percentual de registros ligáveis entre tabelas e possíveis vieses introduzidos pelo vínculo.

**Saídas:** tabela de coorte, fluxograma, histograma de documentos por paciente e série temporal mensal.

### 3.2 Auditoria dos rótulos

Calcular:

- contagem e prevalência das três classes;
- razão de desbalanceamento `maior classe / menor classe`;
- prevalência por instituição, período, setor, tipo de nota, faixa etária, sexo e demais grupos clinicamente relevantes;
- matriz de confusão entre anotadores ou entre rótulo operacional e rótulo adjudicado;
- concordância percentual e **kappa ponderado** para dois avaliadores;
- **alfa de Krippendorff ordinal** quando houver mais avaliadores, dados ausentes ou desenho mais complexo;
- taxa de rótulos ausentes, incertos ou corrigidos;
- concordância dos rótulos com variáveis usadas apenas para validação clínica, como internação/UTI, sem transformar esses desfechos posteriores em preditores.

Para escalas de triagem, a concordância entre avaliadores pode variar substancialmente. Portanto, não trate o rótulo histórico como verdade perfeita. Faça dupla anotação estratificada de uma amostra, adjudicação por especialista e análise dos casos discordantes.

Se cinco níveis forem condensados em três, publique a tabela de mapeamento e faça uma análise de sensibilidade com mapeamentos alternativos clinicamente plausíveis.

### 3.3 Completude, formato e qualidade do texto

Por classe e fonte, medir:

- texto ausente, vazio, somente espaços ou extremamente curto;
- caracteres, palavras, sentenças e tokens do tokenizer que será usado;
- mediana, IQR, P95/P99 e outliers de comprimento;
- percentual que seria truncado em 128, 256, 512 e no limite real do modelo;
- idioma e mistura de idiomas;
- caracteres inválidos, problemas de codificação e artefatos de OCR/digitação;
- abreviações, siglas, unidades, números e faixas de referência;
- marcadores de desidentificação;
- presença e ordem de seções: queixa, história, exame, achados, conclusão, conduta etc.;
- vocabulário, frequência de termos raros e fragmentação em subpalavras.

Como a linha principal usa TF-IDF, acrescentar:

- número de features para cada combinação de `min_df`, `max_df`, `ngram_range` e `max_features`;
- número e percentual de elementos não zero (`nnz`) da matriz;
- densidade/esparsidade global e por documento;
- mediana, IQR e P95 de features ativas por documento;
- distribuição do document frequency e dos valores de IDF;
- proporção de features que aparecem em pouquíssimos documentos;
- estabilidade dos principais n-gramas entre folds, períodos e instituições;
- uso de memória estimado para CSR/CSC e para eventuais conversões densas;
- cobertura e sobreposição entre vocabulários de treino, validação temporal e site externo.

**Gráficos:** violin/boxplot do número de tokens por classe e fonte; histograma de comprimento; matriz de presença de seções; tabela de taxa de truncamento.

Não remova indiscriminadamente stopwords, números, pontuação ou negações. Em texto clínico, `sem`, `nega`, `não`, medidas, doses e unidades podem inverter ou determinar o sentido. Preserve `raw_text` e gere uma versão normalizada separada, com transformações rastreáveis.

### 3.4 Duplicatas, texto copiado e templates

Executar antes da divisão dos dados:

- hash do texto normalizado para duplicatas exatas;
- detecção de quase duplicatas com MinHash/LSH, SimHash ou similaridade de n-gramas;
- investigação de seções copiadas de atendimentos anteriores;
- frequência de templates/boilerplate por classe, instituição e autor;
- duplicatas com rótulos conflitantes;
- pares semelhantes localizados em pacientes diferentes;
- matriz de duplicatas candidatas entre futuros conjuntos de treino, validação e teste.

Relatar:

- `% duplicatas exatas`;
- `% quase duplicatas` em limiar documentado;
- `% conflitos de rótulo entre duplicatas`;
- `% de pacientes com mais de um documento`;
- quantidade de clusters de templates e sua associação com o rótulo.

### 3.5 Auditoria de vazamento de alvo e de tempo

Crie um dicionário `campo → timestamp de disponibilidade → permitido?` e verifique:

- menção explícita ao nível de triagem, cor ou prioridade;
- cabeçalhos, códigos, filas ou nomes de setores que codificam a classe;
- diagnóstico final, alta, internação, UTI, óbito ou procedimentos posteriores ao momento zero;
- frases de comunicação do resultado que tenham sido usadas para produzir o próprio rótulo;
- metadados de autor ou hospital que funcionem como atalho;
- informação derivada usando o conjunto completo antes da divisão: TF-IDF, vocabulário, imputação, seleção de atributos, PCA/UMAP ou normalização.

Faça três baselines diagnósticos:

1. modelo apenas com metadados não clínicos, como fonte/autor/template;
2. modelo apenas com frases suspeitas de entregar o rótulo;
3. modelo com essas informações removidas.

Uma queda muito grande entre eles indica que o sistema pode estar aprendendo o processo de documentação, e não a urgência clínica.

### 3.6 Sinais linguísticos e clínicos por classe

Após concluir as auditorias anteriores, analisar:

- unigramas, bigramas e trigramas mais discriminativos por classe;
- log-odds com prior informativo, qui-quadrado ou informação mútua, sempre com frequência mínima;
- distribuição de negação, incerteza, hipótese, histórico familiar e temporalidade;
- entidades clínicas: sintomas, sinais, anatomia, doenças, medicamentos, procedimentos, medidas e dispositivos;
- conceitos por seção, não apenas no documento inteiro;
- comprimento, densidade de números e de conceitos por classe;
- tópicos/clusters com TF-IDF+NMF ou BERTopic, usados como exploração, não como prova clínica;
- projeção UMAP de embeddings colorida alternadamente por classe, instituição, tipo de nota, autor e período.

Prefira tabelas de termos discriminativos com exemplos **desidentificados e aprovados**, em vez de word clouds. A visualização de embeddings é útil para revelar clusters de instituição/template, mas não demonstra separabilidade nem qualidade do futuro classificador.

Para textos em português, compare a fragmentação do tokenizer de um modelo geral em português e de um modelo clínico. BioBERTpt é uma referência brasileira para representações clínicas; BERTimbau é um baseline geral em português. Para um baseline robusto a abreviações e erros de digitação, use também TF-IDF de caracteres.

### 3.7 Heterogeneidade, subgrupos e drift

Analisar distribuição de classe, missingness, comprimento, vocabulário e conceitos por:

- instituição e setor;
- tipo de documento e especialidade;
- autor ou grupo profissional;
- mês/ano e mudanças de sistema;
- idade, sexo e grupos demográficos disponíveis e eticamente apropriados;
- idioma;
- modalidade/região anatômica, se forem laudos.

Usar:

- qui-quadrado + Cramér's V para associações categóricas;
- Kruskal-Wallis + tamanho de efeito para comprimento/densidade;
- Jensen-Shannon ou PSI para drift de distribuição;
- intervalos de confiança e supressão de células pequenas.

Com amostras grandes, não interprete apenas p-valores: reporte tamanho de efeito e relevância clínica.

### 3.8 Planejamento da divisão treino/validação/teste

Ordem recomendada:

1. resolver duplicatas e definir a unidade observacional;
2. agrupar todos os documentos do mesmo paciente no mesmo conjunto;
3. usar uma divisão temporal realista: período antigo para treino, intermediário para validação e mais recente para teste;
4. se houver várias instituições, reservar uma para validação externa ou usar validação interna-externa por site;
5. ajustar vocabulário, tokenizer adicional, imputação, seleção, TF-IDF e qualquer transformação somente no treino;
6. usar validação cruzada agrupada por paciente dentro do desenvolvimento;
7. manter o teste final selado.

A EDA de integridade pode olhar o conjunto completo para detectar duplicação e problemas de coorte, mas análises que usam o rótulo para escolher features, regras ou hiperparâmetros devem ser feitas no conjunto de desenvolvimento.

## 4. Métricas recomendadas

### 4.1 Métricas da EDA

| Dimensão | Métricas principais |
|---|---|
| Volume | pacientes, atendimentos, notas, notas/paciente, período, fontes |
| Rótulo | prevalência, razão de desbalanceamento, entropia, ausentes/incertos |
| Anotação | concordância, kappa ponderado, alfa de Krippendorff ordinal |
| Texto | mediana/IQR/P95 de caracteres, palavras e tokens; taxa de truncamento |
| Vocabulário | termos únicos, termos raros, cobertura e fragmentação em subpalavras |
| Qualidade | texto vazio, idioma inesperado, artefatos, seções ausentes |
| Redundância | duplicatas exatas, quase duplicatas, conflitos de rótulo |
| Drift | Jensen-Shannon/PSI por tempo, site e tipo de nota |
| Associação | Cramér's V, tamanho de efeito, log-odds/MI dos termos |

### 4.2 Métricas do futuro modelo

Não escolha um único número. Para três classes ordenadas, use o painel abaixo:

| Prioridade | Métrica | Por que usar |
|---|---|---|
| Primária | Macro-F1 | Dá o mesmo peso às três classes |
| Primária de segurança | Recall/sensibilidade da classe alta | Mede quantos urgentes foram encontrados |
| Primária de segurança | Subtriagem grave `alta → baixa` | Mede diretamente o erro mais perigoso |
| Primária ordinal | MAE ordinal e macro-MAE | Considera a distância entre classes; macro-MAE reduz efeito do desbalanceamento |
| Primária ordinal | Kappa ponderado quadrático | Mede concordância corrigida pelo acaso e penaliza mais erros distantes |
| Secundária | Precisão, recall e F1 por classe | Revela onde o modelo falha |
| Secundária | Balanced accuracy | Média dos recalls; melhor que acurácia sob desbalanceamento |
| Secundária | Matriz de confusão normalizada | Mostra subtriagem e supertriagem |
| Secundária | AUPRC classe alta vs. demais e macro-AUPRC | Informativa para classe rara |
| Complementar | ROC-AUC one-vs-rest | Facilita comparação com a literatura, mas não deve ficar isolada |
| Probabilidade | Log loss, Brier multiclasses, curvas de calibração e ECE | Verifica se probabilidades podem sustentar decisões |
| Utilidade | Curva de decisão/net benefit | Avalia benefício clínico em limiares de ação |
| Incerteza | IC 95% por bootstrap agrupado por paciente | Evita falsa precisão com notas repetidas |

Defina com especialistas uma matriz de custo. Exemplo conceitual: `alta → baixa` recebe custo maior; `baixa → alta` representa supertriagem e custo de recurso; erros adjacentes recebem custos intermediários. Não invente limiares aceitáveis sem validação clínica.

### 4.3 Diagnósticos específicos de TF-IDF + árvores

| Diagnóstico | O que reportar |
|---|---|
| Dimensionalidade | número de linhas, colunas e features após cada configuração |
| Esparsidade | `nnz`, densidade e features ativas por documento |
| Raridade | distribuição de document frequency e proporção de termos raros |
| Estabilidade | Jaccard ou rank correlation dos top n-gramas entre folds/tempo/site |
| Memória/tempo | tempo de vetorização, treino, inferência e pico de memória |
| Ganho do SVD | variância explicada, desempenho e custo para diferentes dimensões |
| Importância | estabilidade da importância/permutation importance em reamostragens |
| Generalização | perda de cobertura e desempenho em período/site não visto |

Árvores tradicionais podem ter dificuldade para explorar dezenas de milhares de colunas esparsas. Isso não invalida a estratégia, mas torna obrigatória a comparação entre a matriz TF-IDF bruta e uma representação reduzida por TruncatedSVD. Nunca use PCA convencional na matriz esparsa se ele exigir densificação.

## 5. Técnicas que mais se adequam ao problema

### Essenciais

- divisão temporal e agrupada por paciente;
- detecção de duplicatas e templates;
- auditoria de disponibilidade temporal das features;
- análise de rótulo e concordância entre avaliadores;
- TF-IDF de palavras e caracteres + modelos de árvore como linha principal;
- regressão logística/SVM linear apenas como controle de sanidade;
- comparação entre TF-IDF esparso bruto e TF-IDF reduzido por TruncatedSVD;
- comparação nominal versus ordinal;
- ponderação de classes ou perdas sensíveis a custo;
- modelo de linguagem compatível com idioma e domínio;
- ablação por seções e por fonte de dados;
- calibração e análise de subgrupos;
- validação temporal e, quando possível, externa.

### Úteis, mas secundárias

- NER/conceitos clínicos com negação e temporalidade;
- NMF/BERTopic para descobrir temas e artefatos;
- UMAP para investigar clusters de fonte/template;
- SHAP ou importância de n-gramas para auditoria de atalhos;
- active learning para priorizar exemplos a serem revisados por especialistas.

### Cuidados

- **SMOTE:** aparece com frequência na literatura de triagem, mas não deve ser aplicado ao texto bruto, antes da divisão, nem mecanicamente sobre TF-IDF. Priorize pesos de classe, focal loss ou amostragem somente dentro do treino; compare por validação.
- **Árvores em TF-IDF:** Random Forest e Extra Trees podem ficar caros em memória/tempo com vocabulários muito grandes. XGBoost e LightGBM costumam ser opções mais práticas para matrizes esparsas, mas ainda precisam de regularização e comparação com SVD.
- **Importância de features:** ganho/impureza pode favorecer features ou divisões específicas e ficar instável entre amostras. Confirme com permutation importance no conjunto de validação e estabilidade por bootstrap/folds.
- **CatBoost:** sua capacidade nativa de processar texto é uma rota diferente. Se a entrada for TF-IDF externo, trate-o como modelo de árvore sobre features numéricas e não misture os dois experimentos.
- **Remoção de stopwords:** pode apagar negação e relações clínicas.
- **LLM gerando exemplos sintéticos:** pode reforçar estereótipos, vazar padrões e produzir textos clinicamente artificiais; mantenha experimento separado e teste apenas em dados reais.
- **Explicabilidade:** atenção ou SHAP não provam causalidade; use-os para auditoria e revisão clínica.
- **Notas posteriores:** não use sumários de alta ou diagnósticos finais para prever urgência na chegada.

## 6. Baselines e sequência experimental após a EDA

1. Dummy majoritário e dummy estratificado.
2. Controle de sanidade: TF-IDF de palavras/caracteres + regressão logística ou SVM linear.
3. Linha principal A: TF-IDF esparso + Random Forest/Extra Trees em configuração controlada.
4. Linha principal B: TF-IDF esparso + XGBoost ou LightGBM multiclasses.
5. Linha principal C: TF-IDF + TruncatedSVD + modelos de árvore, comparando dimensões como 100, 300 e 500 sem fixá-las antecipadamente.
6. Variante ordinal: dois classificadores cumulativos (`P(y ≥ média)` e `P(y ≥ alta)`) ou outro método ordinal compatível, com correção de probabilidades incoerentes.
7. Texto + variáveis estruturadas disponíveis no momento zero.
8. Ablations: palavras versus caracteres; somente queixa; somente seções iniciais; sem boilerplate; sem termos suspeitos; sem metadados de fonte.

No TF-IDF, ajustar dentro dos folds apenas: `analyzer`, `ngram_range`, `min_df`, `max_df`, `max_features`, `sublinear_tf` e normalização. Nos modelos de árvore, priorizar busca controlada de profundidade, número de árvores, learning rate quando aplicável, tamanho mínimo de folha, amostragem de linhas/colunas e regularização. Pesos de classe ou `sample_weight` devem ser calculados somente a partir do treino de cada fold.

Todos devem usar exatamente os mesmos pacientes e períodos de validação. Compare com intervalos de confiança e teste de significância apropriado para previsões pareadas.

## 7. Observação sobre MIMIC

- **MIMIC-IV-ED** contém identificadores de paciente/atendimento, queixa principal, acuity/triage e variáveis estruturadas; é adequado para protótipos de triagem, mas a “queixa principal” é bem mais curta que um prontuário completo.
- **MIMIC-IV-Note** contém sumários de alta e laudos radiológicos. Ligar um sumário de alta ao rótulo de triagem para simular uma previsão na chegada introduz informação posterior ao momento zero.
- Para laudos, MIMIC-IV-Note é útil como corpus, mas um rótulo real de urgência/comunicação precisa ser identificado ou anotado; não se deve inferir que o diagnóstico final seja automaticamente a urgência.

## 8. Roteiro pronto para passar a outra IA

Copie o bloco abaixo e substitua os campos entre colchetes.

```text
Você é um cientista de dados sênior com experiência em NLP clínico, epidemiologia e avaliação de modelos de risco. Faça uma análise exploratória reprodutível de um dataset de textos clínicos para classificação ordinal em três níveis de urgência: BAIXA < MÉDIA < ALTA.

CONTEXTO
- Arquivo(s): [CAMINHO OU DESCRIÇÃO]
- Idioma principal: [PORTUGUÊS/INGLÊS/OUTRO]
- Coluna de texto: [TEXT_COL]
- Coluna alvo: [TARGET_COL]
- Ordem das classes: [BAIXA, MÉDIA, ALTA]
- ID do paciente: [PATIENT_ID]
- ID do atendimento: [ENCOUNTER_ID]
- ID do documento: [DOCUMENT_ID]
- Data/hora do texto: [TEXT_TIME]
- Data/hora da decisão: [PREDICTION_TIME]
- Instituição/setor: [SITE_COL]
- Tipo de documento/seção: [NOTE_TYPE/SECTION_COL]
- Demográficos disponíveis: [LISTA]
- Objetivo clínico: [URGÊNCIA NA CHEGADA OU URGÊNCIA DE COMUNICAÇÃO DO LAUDO]
- Escala/rótulo original e regra de conversão para 3 classes: [DESCREVER]
- Estratégia principal: TF-IDF + modelos de árvore [RANDOM FOREST/EXTRA TREES/XGBOOST/LIGHTGBM/OUTROS]

REGRAS OBRIGATÓRIAS
1. Não altere o arquivo-fonte.
2. Não exiba prontuários completos nem informações identificáveis. Exemplos textuais devem ser desidentificados ou substituídos por padrões agregados.
3. Antes de analisar palavras, confirme unidade observacional, proveniência, momento zero, disponibilidade temporal das colunas e qualidade do rótulo.
4. Trate o alvo como ordinal. Um erro ALTA→BAIXA é mais grave que ALTA→MÉDIA.
5. Detecte duplicatas antes de propor a divisão dos dados.
6. Nenhum paciente pode aparecer em mais de um split.
7. Prefira divisão temporal; mantenha o teste final selado.
8. Ajuste TF-IDF, vocabulário, imputação, seleção de features, PCA/UMAP e qualquer transformação apenas no treino.
9. Não remova negações, números, unidades, seções ou pontuação clinicamente relevante sem um experimento de ablação.
10. Não aplique SMOTE ao texto bruto nem antes do split.
11. Se faltar alguma coluna, registre a limitação e execute tudo que ainda for válido; não invente dados.
12. Não treine um modelo complexo antes de concluir e resumir a EDA.

FASE A — DICIONÁRIO E COORTE
- Liste schema, tipos, chaves, cardinalidade, missingness e exemplos apenas desidentificados.
- Calcule pacientes, atendimentos, documentos, documentos/paciente, período, sites, setores e tipos de nota.
- Produza tabela/fluxograma de inclusão e exclusão.
- Crie tabela: feature, origem, timestamp de disponibilidade, uso permitido no momento da previsão, risco de vazamento.

FASE B — RÓTULOS
- Conte e mostre a prevalência das classes; calcule razão de desbalanceamento e entropia.
- Analise rótulo por tempo, site, tipo de nota e subgrupos.
- Se existirem múltiplos anotadores, calcule concordância e kappa ponderado; para desenho ordinal com múltiplos avaliadores, use alfa de Krippendorff ordinal.
- Liste rótulos ausentes, incertos e conflitos entre duplicatas.
- Se a escala original foi agrupada em três níveis, teste a sensibilidade a mapeamentos alternativos fornecidos; não invente nova regra clínica.

FASE C — QUALIDADE E COMPRIMENTO DO TEXTO
- Meça vazio/curto, caracteres, palavras, sentenças e tokens por classe/fonte.
- Reporte mediana, IQR, P95 e P99.
- Calcule taxa de truncamento para 128, 256, 512 e o limite do tokenizer escolhido.
- Detecte idioma, artefatos de codificação/OCR, marcadores de desidentificação, abreviações, números/unidades e seções.
- Preserve raw_text; crie clean_text apenas com normalização rastreável.
- Para cada configuração candidata de TF-IDF, reporte vocabulário, shape, nnz, densidade, features ativas/documento, distribuição de IDF, memória e cobertura no período/site futuro.
- Avalie a estabilidade dos top n-gramas entre folds, tempo e site.

FASE D — DUPLICATAS E VAZAMENTO
- Encontre duplicatas exatas por hash normalizado e quase duplicatas com método escalável.
- Agrupe templates/boilerplate e meça associação com classe/site/autor.
- Procure termos/códigos que revelem diretamente prioridade, cor, ESI/MTS, desfecho, UTI, alta, óbito, diagnóstico final ou condutas posteriores.
- Verifique texto copiado entre visitas do mesmo paciente.
- Gere uma tabela de riscos e ações: remover, restringir a seção, manter com justificativa ou revisar clinicamente.

FASE E — LINGUAGEM E CONTEÚDO CLÍNICO
- Produza unigramas/bigramas/trigramas mais discriminativos por classe usando log-odds com prior, qui-quadrado ou informação mútua; aplique frequência mínima.
- Analise negação, incerteza, temporalidade, histórico familiar e experiencer, se houver ferramenta adequada ao idioma.
- Extraia entidades/conceitos clínicos quando houver modelo validado no idioma.
- Faça TF-IDF+NMF ou BERTopic apenas como exploração.
- Faça UMAP de embeddings e colore separadamente por classe, site, tipo de nota, autor e período; interprete clusters de fonte/template, não como prova de desempenho.
- Para português, compare fragmentação de tokenizer geral e clínico; inclua TF-IDF de caracteres como baseline futuro.

FASE F — HETEROGENEIDADE E DRIFT
- Compare distribuição de classes, missingness, comprimento e vocabulário por tempo/site/tipo/subgrupo.
- Use qui-quadrado+Cramér's V para categóricas, Kruskal-Wallis+tamanho de efeito para contínuas e Jensen-Shannon/PSI para drift.
- Reporte tamanho de efeito e IC, não apenas p-valor. Suprima células pequenas.

FASE G — PROPOSTA DE SPLIT
- Proponha treino/validação/teste agrupado por PATIENT_ID e ordenado por tempo.
- Se houver múltiplos sites, proponha teste externo por site.
- Gere tabela de pacientes, notas, prevalência e comprimento por split.
- Prove que não há paciente nem duplicata/quase duplicata cruzando os splits.
- Não use o teste para escolher limpeza, features ou hiperparâmetros.

FASE H — PLANO DE BASELINES E MÉTRICAS
- Controle de sanidade: dummy e TF-IDF+logística/SVM linear. Não os trate como a estratégia principal.
- Linha principal A: TF-IDF esparso de palavras e caracteres + Random Forest/Extra Trees em configuração viável.
- Linha principal B: TF-IDF esparso + XGBoost/LightGBM multiclasses.
- Linha principal C: TF-IDF + TruncatedSVD + os mesmos modelos de árvore. Compare dimensões, variância explicada, memória, tempo e desempenho.
- Variante ordinal: compare multiclasses nominal com classificadores cumulativos para `y ≥ média` e `y ≥ alta` ou outro método ordinal compatível.
- Ajuste o TF-IDF integralmente dentro de cada fold. Nunca construa o vocabulário no dataset completo.
- Use pesos de classe/sample_weight somente no treino do fold. Não use SMOTE no texto bruto nem antes do split.
- Métricas obrigatórias: matriz de confusão, precision/recall/F1 por classe, macro-F1, balanced accuracy, recall de ALTA, taxa ALTA→BAIXA, MAE ordinal, macro-MAE, kappa ponderado quadrático, AUPRC de ALTA e macro-AUPRC, ROC-AUC OVR, log loss/Brier multiclasses e curvas de calibração.
- Calcule IC 95% com bootstrap agrupado por paciente.
- Proponha matriz de custo para subtriagem/supertriagem, mas deixe os valores para validação clínica.

ENTREGÁVEIS
1. Notebook ou script modular e reproduzível.
2. data_dictionary.csv.
3. cohort_summary.csv.
4. label_audit.csv.
5. leakage_audit.csv.
6. duplicate_clusters.csv sem texto identificável.
7. pasta de gráficos em PNG/SVG.
8. EDA_REPORT.md com: resumo executivo, método, resultados, riscos, decisões recomendadas e limitações.
9. Checklist final PASS/WARN/FAIL para: rótulo, tempo, duplicatas, split por paciente, desbalanceamento, drift, privacidade e prontidão para modelagem.

CRITÉRIO DE ACEITAÇÃO
Termine com uma conclusão objetiva: (A) pronto para modelagem, (B) pronto somente após correções listadas ou (C) ainda inadequado. Para cada bloqueio, informe evidência, impacto e ação recomendada. Não faça afirmações clínicas causais a partir de associações exploratórias.
```

## 9. Leituras que sustentam o roteiro

1. Porto BM. *Improving triage performance in emergency departments using machine learning and natural language processing: a systematic review* (2024). https://link.springer.com/article/10.1186/s12873-024-01135-2
2. Spasic I, Nenadic G. *Clinical Text Data in Machine Learning: Systematic Review* (2020). https://medinform.jmir.org/2020/3/e17984/
3. Tiwari A et al. *Automatic Classification of Critical Findings in Radiology Reports* (2017). https://proceedings.mlr.press/v69/tiwari17a/tiwari17a.pdf
4. Banerjee I et al. *Natural Language Processing Model for Identifying Critical Findings—A Multi-Institutional Study* (2023). https://pmc.ncbi.nlm.nih.gov/articles/PMC9984612/
5. Meng X et al. *Self-Supervised Contextual Language Representation of Radiology Reports to Improve the Identification of Communication Urgency* (2020). https://pmc.ncbi.nlm.nih.gov/articles/PMC7233055/
6. Casey A et al. *A systematic review of natural language processing applied to radiology reports* (2021). https://link.springer.com/article/10.1186/s12911-021-01533-7
7. Collins GS et al. *TRIPOD+AI statement* (2024). https://www.bmj.com/content/385/bmj-2023-078378
8. Riley RD et al. *Evaluation of clinical prediction models (part 2)* (2024). https://pmc.ncbi.nlm.nih.gov/articles/PMC10788734/
9. Rosenblatt M et al. *Data leakage inflates prediction performance in connectome-based machine learning models* (2024). https://www.nature.com/articles/s41467-024-46150-w
10. French S et al. *Assessment of Interrater Reliability of the Emergency Severity Index After Implementation in Jamaica* (2021). https://pubmed.ncbi.nlm.nih.gov/33097242/
11. Sakai T. *Evaluating Evaluation Measures for Ordinal Classification and Ordinal Quantification* (2021). https://aclanthology.org/2021.acl-long.214.pdf
12. Schneider ETR et al. *BioBERTpt — A Portuguese Neural Language Model for Clinical Named Entity Recognition* (2020). https://aclanthology.org/2020.clinicalnlp-1.7/
13. MIMIC-IV-ED. https://physionet.org/content/mimic-iv-ed/
14. MIMIC-IV-Note. https://physionet.org/content/mimic-iv-note/
