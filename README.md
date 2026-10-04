# Experimentos de recuperação de documentos científicos

**Comece pelo [guia passo a passo para executar os experimentos](docs/como_executar.md).**
Ele cobre instalação, A100 no Colab Pro, progresso, resultados, retomada e os
dois notebooks do estudo anterior.

Os estudos estão separados em três fluxos:

| Experimento | Entrada | Execução |
|---|---|---|
| Comparação BM25 versus BM25+BGE | Datasets BEIR | [Notebook original](scifact_bm25_reranker_colab.ipynb) |
| Tentativa de reranker seletivo | CSVs da comparação anterior | [Notebook seletivo](notebooks/reranker_seletivo.ipynb) |
| TCC: expansão seletiva de consultas | Datasets BEIR, independentemente do estudo anterior | [Notebook de expansão](notebooks/query_expansion.ipynb) |

O reranker seletivo decide **se aplica BGE**. A expansão seletiva decide **se
expande a consulta**; os caminhos B–F terminam com BGE.

Instalação local, na raiz do projeto:

```bash
python -m pip install -e ".[notebooks]"
python -m qexp.cli run --config configs/smoke.json
```

O primeiro uso baixa datasets e modelos. Para todas as queries, use
`configs/full.json`. As configurações padrão são pontos de partida: fixe as
revisões dos modelos antes de uma execução final.

Consulte [o guia de execução](docs/experiments.md) para Colab, etapas, retomada,
outputs e reutilização dos resultados anteriores; o [protocolo do TCC](docs/query_expansion.md)
descreve pipelines, métricas e hipóteses.

Testes locais, sem baixar modelos:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Para executar a expansão remotamente na GPU da sua conta Colab, pelo terminal:

```bash
python -m pip install -e ".[colab]"
python -m qexp.colab run --config configs/smoke.json --gpu A100
```

A primeira execução solicita autenticação Google pela CLI oficial. O comando
usa unidades de computação da conta, baixa os resultados e encerra a sessão
criada após confirmar o download. Use `--dry-run` para simular sem alocar GPU.
Veja [acesso remoto ao Colab](docs/colab.md) para retomada e recuperação de falhas.

A execução mostra progresso automaticamente com Rich: etapa atual, dataset/query,
porcentagem, tempo e checkpoints reutilizados. No Colab, as atualizações aparecem
no terminal e em `experiment.log`; `qexp_progress.json` registra o último estado.
