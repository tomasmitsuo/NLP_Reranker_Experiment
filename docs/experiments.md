# Execução dos experimentos separados

Para começar do zero, consulte o [guia passo a passo de execução](como_executar.md).

## 1. Comparação anterior: BM25 versus BGE

`scifact_bm25_reranker_colab.ipynb` mantém recuperação, avaliação, análises
qualitativas, RRF, gráficos e opções de busca densa/sweep do estudo anterior.
As células de expansão e classificadores foram removidas desse notebook.
Os métodos e as métricas anteriores foram preservados.

Execute no Colab como antes. A primeira célula cria uma pasta exclusiva em
`results/baseline/<data-hora>/` e muda o diretório de trabalho para ela. Os CSVs
são salvos ali; a organização final os coloca na subpasta `results/` dessa
execução. O ZIP contém somente essa execução. Guarde o caminho impresso.
Para uma execução mais curta, desative o sweep e pule as visualizações 3D;
ainda não há checkpoints por query no notebook histórico.

## 2. Tentativa de reranker seletivo

Abra `notebooks/reranker_seletivo.ipynb`. Defina `BASELINE_RESULTS` como a pasta
de uma execução específica do estudo anterior (com ou sem organização final).
Por padrão, aponta para os resultados existentes em `PLN_results_outputs/`.
Precisa apenas de:

- `rankings_all_datasets.csv`, incluindo `doc_text`;
- `results_all_datasets.csv`, incluindo métricas e `elapsed_sec` de ambos os pipelines.

O leitor busca esses dois arquivos recursivamente e rejeita múltiplas execuções
misturadas ou queries não pareadas. Os IDs são lidos como strings.
Não executa BM25/BGE nem carrega os corpora. Treina os classificadores a partir
dos rankings salvos; o benchmark query-only ainda gera embeddings das queries.
As funções estão em `src/pln_legacy/selective.py`.

Os folds, features, modelos e análise de custo são os do estudo anterior.
Use múltiplos datasets e amostras suficientes das duas classes para o protocolo
de cinco folds e leave-one-dataset-out; três queries de smoke não bastam.
O benchmark da MLP também requer exemplos suficientes por fold para sua
validação interna. As limitações metodológicas do estudo não foram alteradas.

Cada execução grava em `results/reranker_seletivo/<data-hora>/`, sem substituir
os CSVs de entrada. As células de treino, benchmark query-only e auditoria
continuam separadas para permitir executar apenas o necessário.

## 3. TCC: expansão seletiva de consultas

O pacote `src/qexp/` não importa os classificadores nem depende de executar
os baselines históricos. Recalcula A/B no protocolo do TCC para comparar C–F.

```bash
python -m pip install -e ".[notebooks]"
python -m qexp.cli prepare --config configs/smoke.json
python -m qexp.cli generate --config configs/smoke.json
python -m qexp.cli run --config configs/smoke.json
python -m qexp.cli analyze --config configs/smoke.json
```

`prepare` e `generate` são opcionais: `run` resolve dependências ausentes.
Executar `generate` em um processo separado permite liberar a memória do LLM
antes de executar o BGE com as expansões já em cache.
`analyze` usa o snapshot salvo, inclusive sua configuração original, sem
carregar dados/modelos ou consultar o cache de gerações.

Use `configs/full.json` para todos os datasets e queries. Os caminhos dos JSONs
são relativos a `project_root`, resolvido a partir da pasta da configuração;
não dependem do diretório corrente. Crie outro JSON/output_dir para uma nova
variante que queira preservar lado a lado.

### Retomada e reutilização

- Corpus/índice: `.cache/prepared/`, por dataset, limite de queries, loader,
  tokenização, versão das bibliotecas e parâmetros BM25. São arquivos pickle
  locais; use apenas caches gerados pelo próprio projeto. Para atualizar o
  conteúdo de uma fonte remota com o mesmo ID, use um diretório de cache novo.
- Expansões: `.cache/query_expansion/`, por prompt/modelo/revisão/decodificação.
- Queries completas: `.cache/qexp_checkpoints/`, por código, configuração,
  versões, conteúdo do corpus, query, julgamentos e parâmetros do índice.
- Resultados consolidados: `results/query_expansion/{smoke,full}/`. O snapshot
  `qexp_run.json` permite refazer análises sem inferência.

Uma interrupção preserva queries concluídas; a query em andamento é refeita.
Reexecute o mesmo comando para retomar. D/E/E_RA reutilizam o mesmo ranking
Multi-query por query, sem repetir o BGE para caminhos equivalentes. Cada
política continua recebendo os custos dos estágios que exigiria.

A retomada usa checkpoints da query inteira: mudar limiares recalcula os
rankings dessa query, embora as gerações válidas continuem em cache. Ainda não
há cache persistente independente de cada ranking B/D para varrer limiares.
Ao usar geradores/rerankers personalizados pela API Python, altere também
`checkpoint_tag` para identificar a implementação injetada.

Os custos de checkpoints são históricos, identificados no metadata. Para
medir novamente, use `--no-resume` e um novo `cache_dir` de geração. Isso não
converte custos compostos em medições independentes de latência ponta a ponta.
Fixe commits de LLM/BGE: `main` pode mudar sem invalidar uma chave de cache.

### Colab

Também é possível enviar o experimento pelo terminal usando a CLI oficial,
sem copiar manualmente o projeto para o Drive. Consulte [execução remota](colab.md).


Disponibilize o projeto completo no Drive (notebooks, `src/`, `configs/` e
`pyproject.toml`), monte-o e ajuste `PROJECT_ROOT` no notebook. Execute a célula
de instalação. Os notebooks novos não são arquivos autocontidos: importam o
pacote para manter uma única implementação testável.

Em `query_expansion.ipynb`, use `PROFILE="smoke"`/`"full"` e
`STAGE="prepare"`/`"generate"`/`"run"`/`"analyze"`. Os gráficos leem os CSVs
salvos e podem ser refeitos sem inferência. O ZIP é gerado apenas para a pasta
do experimento; no Colab, `google.colab.files.download(archive)` faz o download.

## Organização do código

`datasets.py` prepara dados; `retrieval.py` contém BM25/RRF/RM3;
`generation.py` contém LLM/cache; `reranking.py` encapsula BGE;
`selection.py` calcula sinais/regras; `runner.py` compõe pipelines;
`checkpoints.py` salva/retoma queries; `metrics.py` e `evaluation.py`
calculam métricas e hipóteses. A CLI e os notebooks usam esses mesmos módulos.

Esta reorganização preserva H1–H4. Controles adicionais discutidos na revisão
(seleção aleatória com orçamento equivalente, comparação E_RA–E e controle de
truncamento em H4) não foram acrescentados nesta alteração.

## Acompanhamento do progresso

A expansão usa [Rich](https://rich.readthedocs.io/en/stable/progress.html),
instalado como dependência do projeto. O acompanhamento é automático no
notebook e na CLI, sem conta ou serviço externo.

Em terminais interativos/notebooks, há barra de queries, porcentagem, tempo
transcorrido e estimativa de tempo restante. A etapa atual identifica dataset,
query e pipeline (A/B/C/C_H4/D/E/E_RA/F), incluindo carga de modelos,
preparação dos dados, geração e exportação. Antes de conhecer o total de
queries, mostra a etapa em andamento sem inventar uma porcentagem.

No stream do Colab e em arquivos redirecionados, usa mensagens de texto:

```text
[progresso 120s] queries 1/3 (33.3%) | novas=1 cache=0 | SciFact | 3 | B: reranqueamento BGE original
```

O contador global não reinicia em cada query. Uma query de `run` só é contada
após carregar seu checkpoint ou salvar todos os pipelines concluídos.
Em `generate`, a contagem termina após as duas tentativas de expansão.
A mensagem final de conclusão aparece depois da exportação; 100% das queries
não significa que o relatório já terminou. A estimativa de tempo é aproximada.

A cada 30 segundos, há atualização de atividade, mesmo em uma etapa longa.
Ela indica que o monitor/processo está ativo; não garante avanço do modelo ou
do download. A quantidade de queries concluídas é o indicador de avanço.

O arquivo `<output_dir>/qexp_progress.json` contém o estado mais recente,
contadores, etapa, horário UTC e status `running`, `completed` ou `failed`.
Em uma falha capturada, preserva a contagem parcial e registra o erro.
Uma interrupção abrupta da VM pode deixar o último estado como `running`.
No Colab, o arquivo fica na VM durante a execução e entra no ZIP de resultados;
as mensagens são transmitidas ao terminal e gravadas em `experiment.log`.

Para controlar a apresentação local:

```bash
python -m qexp.cli run --config configs/smoke.json --progress plain
python -m qexp.cli run --config configs/smoke.json --progress off
```

`plain` força mensagens de texto; `off` desativa o monitor e seu arquivo JSON
(as bibliotecas de download ainda podem imprimir suas próprias mensagens).
No notebook, use `execute_stage(..., progress="plain")` ou `progress="off"`.
O comando remoto `qexp.colab` já transmite o progresso automaticamente.
As alterações passam a valer no próximo envio do código ao Colab; não modificam
uma execução que já está em andamento. Como o código mudou, checkpoints criados
por versões anteriores podem ser invalidados pelo controle de reprodutibilidade.
