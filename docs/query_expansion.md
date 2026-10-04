# Expansão seletiva de consultas — implementação do TCC

O código está em `src/qexp/`, com execução pelo notebook
`notebooks/query_expansion.ipynb` ou pelo terminal. O experimento é independente
da comparação anterior BM25/BGE e do classificador de reranqueamento seletivo.
Consulte [execução dos experimentos](experiments.md) para os três fluxos.

## Pipelines

| ID | Recuperação de candidatos | Reranqueamento |
|---|---|---|
| A | BM25 original | Nenhum |
| B | BM25 original top-100 | BGE com query original |
| C | BM25 com query + pseudo-documento | BGE com query original |
| D | BM25 original + quatro reformulações, fundidos por RRF | BGE com query original |
| E | Regras lexicais decidem entre candidatos originais e Multi-query/RRF | BGE com query original |
| E_RA | E acrescentando divergência BM25–BGE inicial | BGE com query original |
| F | Feedback RM3 e BM25 ponderado | BGE com query original |
| C_H4 | Exatamente os candidatos de C | BGE com query expandida |

Todos os métodos são reavaliados com as mesmas queries e os mesmos índices da
execução de expansão, com limite de 100 candidatos e métricas em @10 por padrão. Os
resultados novos usam o prefixo `qexp_`; não substituem as tabelas antigas.

## Executar

Na raiz do projeto, instale `python -m pip install -e ".[notebooks]"`.
Para verificar com três queries de SciFact:

```bash
python -m qexp.cli run --config configs/smoke.json
```

Para o experimento completo, use `configs/full.json`. `run` prepara os dados,
gera expansões ausentes e salva checkpoints por query. Pode ser interrompido e
retomado com o mesmo comando. Não precisa executar o notebook anterior.

As etapas `prepare`, `generate` e `analyze` também são independentes. A última
lê `qexp_run.json` e refaz as tabelas sem executar os modelos ou carregar corpus.
No notebook novo, selecione `PROFILE` e `STAGE`. No Colab, disponibilize a pasta
do projeto (incluindo `src/` e `configs/`) no Drive e ajuste `PROJECT_ROOT`.

O cache de corpus/índice diferencia o limite de queries: uma execução `smoke`
não restringe uma execução `full`. Saídas ficam em `results/query_expansion/`.
Checkpoints incluem hashes do código, configuração, corpus, queries e qrels;
mudar esses itens invalida os resultados correspondentes. Os qrels entram no
hash apenas para impedir métricas antigas após alterações dos julgamentos.

O LLM padrão é `Qwen/Qwen2.5-1.5B-Instruct`, executado localmente com Transformers,
sem API paga. O primeiro uso baixa os pesos. Modelo, revisão, número de
reformulações, limite de geração e repetição da query são configuráveis. Fixe a
revisão em um commit para a execução final. A escolha deste modelo é um ponto de
partida operacional, não uma conclusão sobre sua qualidade na tarefa.

O cache `.cache/query_expansion` guarda saídas por hash de prompt/modelo/revisão/
decodificação. No Colab, use um caminho no Drive se precisar persistir entre
sessões. Saídas vazias ou listas inválidas acionam fallback para candidatos
originais, registrando erro e tempo; elas não são gravadas como cache válido.
Falhas de download, memória ou execução do modelo interrompem a execução.

## Seleção sem vazamento

E usa OR entre gap relativo baixo, entropia normalizada alta e cobertura lexical
baixa nos primeiros dez documentos. É possível escolher apenas uma regra por
`selector_rule`. Os limiares são heurísticos e fixos; não foram calibrados nos
qrels. Para fazer ajuste, separe desenvolvimento e teste antes da avaliação.

E_RA acrescenta `1 - média(overlap dos prefixos @1...@10)` entre BM25 e BGE. Esse
sinal detecta mudanças de ordem mesmo com os mesmos documentos no top-10. Seu
BGE inicial é contabilizado; se não expandir, o ranking já calculado é reutilizado.
E simples é uma política lexical; a variante E_RA é a que utiliza o reranker na decisão.

As funções de geração, seleção e RM3 não recebem qrels. Os julgamentos aparecem
apenas depois do ranking, para métricas e diagnóstico. Candidate Recall real
não é usado como estimador disponível em produção.

## Métricas, hipóteses e tempo

H1 compara Candidate Recall com A. H2 compara nDCG/MRR com B e mede a associação
entre ganhos de recall dos candidatos e nDCG. H3 compara degradação de E/E_RA e D
em relação a B. H4 compara C e C_H4 com candidatos idênticos. O relatório inclui
ganhos, perdas, empates, bootstrap pareado e Wilcoxon com correção de Holm na
tabela de hipóteses. Correlações de H2 são descritivas e não demonstram causalidade.

`estimated_uncached_sec` soma os estágios necessários a cada pipeline, incluindo
o tempo original de geração guardado no cache. `composed_observed_sec` usa o tempo
de consulta ao cache/geração desta execução. Estágios compartilhados entre métodos
são cobrados em cada política que os exige. D/E/E_RA reaproveitam a mesma geração
e ranking Multi-query dentro de cada query; custos continuam atribuídos a cada
política. Resultados retomados de checkpoints preservam seus tempos históricos. Estes são **custos compostos**, não
medições independentes de latência end-to-end em produção. Setup do LLM/BGE e tempo
total da execução aparecem separadamente em `qexp_metadata.json`.

Para comparar tempo no mesmo hardware, use um cache de geração novo e `--no-resume`. Não misture tempos
históricos de geração em CPU e GPU; o hardware de origem aparece no registro das
gerações. Não há conclusão de ganho de qualidade ou economia antes de executar
os modelos nos datasets completos.

## Artefatos

| Arquivo | Conteúdo |
|---|---|
| `qexp_metrics.csv`, `qexp_summary.csv` | Qualidade, recall dos candidatos, tempo e taxas de expansão |
| `qexp_decisions.csv` | Sinais, decisão e motivos por query/política |
| `qexp_rankings.csv` | Rankings completos de candidatos e top-k final |
| `qexp_generations.json` | Prompts, saídas brutas, cache, erros e tempo de geração |
| `qexp_artifacts.json` | Query do BGE, expansões utilizadas e pesos RM3 |
| `qexp_query_deltas.csv`, `qexp_degradation.csv` | Ganhos/perdas contra B e taxas de degradação |
| `qexp_hypotheses.csv`, `qexp_h2_associations.csv` | Comparações pareadas e associação descritiva |
| `qexp_metadata.json` | Parâmetros, datasets, queries, versões e protocolo de tempo |
| `qexp_report.md`, `qexp_quality_cost.png` | Relatório e comparação visual |

Use os IDs em `qexp_query_deltas.csv` para cruzar as queries que pioraram com
prompts/expansões e documentos recuperados, inspecionando mudanças de tema,
negação, entidades e relevância julgada.

## Relação com os artigos fornecidos

- **Query2doc** motiva C. Esta implementação usa prompt zero-shot e query +
  pseudo-documento, conforme a proposta. Não reproduz o prompt few-shot nem o
  modelo do artigo; o peso da query pode ser estudado com `query_repetitions`.
- **BERT-QE** motiva avaliar a qualidade do sinal de expansão. A seleção de
  chunks em três fases do artigo não está implementada nesta proposta.
- **Expand, Rerank, and Retrieve (EAR)** motiva múltiplas expansões. Aqui elas
  são fundidas por RRF; não se treina o reranker de queries do EAR.
- **When do Generative Query and Document Expansions Fail?** motiva os testes
  de degradação e a comparação da query usada no BGE.

RM3 interpola o modelo unigram da query com um modelo dos documentos de feedback,
conforme a [formulação do Lemur](https://www.lemurproject.org/lemur/IndriQueryLanguage.php).
A implementação usa softmax dos scores BM25 para ponderar documentos, stopwords
do scikit-learn e soma de contribuições BM25 com pesos reais. É uma variante
documentada de RM3, não uma reprodução numérica de Anserini/Pyserini.

## Validação local

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Os testes importam os módulos de produção e usam BM25 e corpus pequeno,
com gerador e reranker simulados. Verificam fusão, seleção, cache, fallback,
pesos RM3, independência dos qrels, candidatos de H4, custos, exportação, retomada e invalidação dos checkpoints. Não
medem qualidade de expansões do LLM nem substituem o experimento com BGE real.
