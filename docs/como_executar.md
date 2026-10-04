# Como executar os experimentos

Este é o roteiro para instalar o projeto, escolher o experimento, acompanhar a
execução e encontrar os resultados. Os comandos de terminal abaixo partem da
raiz do repositório e usam seu ambiente virtual `.venv`.

## 1. Escolha o experimento

| Objetivo | O que executar | Pré-requisito |
|---|---|---|
| Comparar BM25 com BM25+BGE | [Notebook original](../scifact_bm25_reranker_colab.ipynb) | Datasets, baixados pelo notebook |
| Treinar uma política que decide se aplica BGE | [Reranker seletivo](../notebooks/reranker_seletivo.ipynb) | CSVs da comparação anterior |
| Avaliar expansão de consultas do TCC | CLI ou [notebook de expansão](../notebooks/query_expansion.ipynb) | Independente dos dois experimentos anteriores |

A ordem do estudo anterior é **comparação BM25/BGE → reranker seletivo**.
Se você já tem os CSVs em `PLN_results_outputs/`, pode ir diretamente ao seletivo.
A expansão do TCC recalcula seus próprios baselines A/B; não precisa executar
os notebooks anteriores.

Para começar pela expansão na GPU da sua conta Colab Pro, siga as seções 2 e 3.

## 2. Prepare o ambiente local

No seu computador, abra um terminal na pasta do projeto:

```bash
cd /home/tomas-ueda/Documents/ufrgs/PLN
```

Em outra máquina, substitua esse caminho pela pasta onde você colocou o projeto.
Use Python 3.10 ou superior; o ambiente local usado na validação foi Python 3.12.

Se ainda não existir `.venv`, crie-o uma vez:

```bash
python3 -m venv .venv
```

Instale o projeto com os recursos para Colab e notebooks:

```bash
.venv/bin/python -m pip install -e ".[colab,notebooks]"
```

A instalação precisa de internet. Os comandos usam explicitamente o Python do
ambiente virtual; não é necessário ativá-lo com `source`.

Confirme que os dois executores estão disponíveis:

```bash
.venv/bin/python -m qexp.cli --help
.venv/bin/python -m qexp.colab --help
```

## 3. Expansão do TCC na A100 do Colab Pro

### 3.1. Confira os comandos sem executar

```bash
.venv/bin/python -m qexp.colab run \
  --config configs/smoke.json --gpu A100 --dry-run
```

`--dry-run` apenas mostra o que seria executado. Não autentica, não envia
arquivos e não aloca GPU. Ver comandos impressos nessa etapa não significa
que o experimento está rodando.

### 3.2. Rode o teste curto

```bash
.venv/bin/python -m qexp.colab run \
  --config configs/smoke.json --gpu A100
```

Na primeira conexão, a CLI oficial apresenta o fluxo de autenticação Google.
Abra a URL exibida, entre com a conta que tem Colab Pro e forneça o código no
próprio terminal quando solicitado. Não é necessário criar uma chave de API.

O executor solicita uma A100, envia o código e a configuração, instala as
dependências na VM, executa o experimento e baixa resultados/checkpoints.
A A100 é o padrão do projeto; sua alocação depende da disponibilidade e dos
recursos permitidos à conta. Não há troca automática para T4 se ela for recusada.
A execução utiliza unidades de computação da conta Colab.

A configuração `smoke` usa **três queries de SciFact**. É uma verificação de
funcionamento; não serve para tirar conclusões sobre as hipóteses.
O primeiro uso também baixa corpus e modelos na VM, o que pode levar tempo.

### 3.3. Rode o experimento completo

Depois de verificar que o teste curto terminou corretamente:

```bash
.venv/bin/python -m qexp.colab run \
  --config configs/full.json --gpu A100
```

| Configuração | Datasets | Queries |
|---|---|---|
| `configs/smoke.json` | SciFact | Até 3 |
| `configs/full.json` | SciFact, TREC-COVID e NF-Corpus | Todas as queries com julgamentos carregadas pelo protocolo |

Antes de uma execução final, fixe `model_revision` e `reranker_revision` em
commits dos modelos, dentro do objeto `qexp` da configuração. Os padrões usam
`main`. Consulte o [protocolo experimental](query_expansion.md) para os demais
parâmetros e a interpretação de H1–H4.

## 4. Acompanhe a execução

O terminal mostra as etapas e o progresso das queries. Exemplo ilustrativo:

```text
[progresso 120s] queries 1/3 (33.3%) | novas=1 cache=0 | SciFact | 3 | B: reranqueamento BGE original
```

- `1/3`: uma query concluída entre três previstas, contando todos os pipelines.
- `novas=1`: uma query foi executada nesta rodada.
- `cache=0`: nenhuma query completa foi recuperada de checkpoint.
- `SciFact | 3`: dataset e identificador da query atual.
- O texto final informa a etapa atual, como carga de modelos, geração ou BGE.

Em terminais interativos, Rich apresenta barra de progresso e estimativa de
tempo. No stream remoto, as atualizações chegam como linhas de texto. Antes de
conhecer o total de queries, a preparação aparece sem uma porcentagem.

Há mensagens de atividade a cada 30 segundos em etapas longas. Elas indicam que
o monitor/processo continua ativo, mas não garantem avanço do cálculo. Confira
o contador de queries e as mudanças de etapa. Após 100% das queries, ainda pode
haver cálculo de hipóteses, exportação e download.

Para consultar a sessão em outro terminal:

```bash
.venv/bin/colab --auth oauth2 sessions
.venv/bin/colab --auth oauth2 status -s pln-ID
.venv/bin/colab --auth oauth2 usage
```

Substitua `pln-ID` pelo nome da sessão impresso no início. Se usou `--session`,
use o nome que você forneceu. O status confirma a situação da VM; o log informa
o progresso do experimento. `usage` consulta saldo e taxa de consumo da conta.

## 5. Encontre e confira os resultados

Para o teste remoto curto, o destino padrão é:

```text
results/query_expansion/smoke/colab/pln-<id>/
```

Para a execução remota completa:

```text
results/query_expansion/full/colab/pln-<id>/
```

O caminho exato aparece no terminal. Cada envio remoto cria uma pasta própria.

| Arquivo dentro dessa pasta | Uso |
|---|---|
| `remote_status.json` | `returncode: 0` indica sucesso do worker remoto |
| `experiment.log` | Saída da instalação, verificação da GPU e experimento |
| `outputs/qexp_progress.json` | Último estado, etapa, contadores e eventual erro |
| `outputs/qexp_summary.csv` | Métricas médias por dataset/pipeline |
| `outputs/qexp_hypotheses.csv` | Comparações estatísticas H1–H4 |
| `outputs/qexp_metrics.csv` | Métricas por query/pipeline |
| `outputs/qexp_report.md` | Relatório das análises |
| `outputs/qexp_run.json` | Snapshot para refazer análises sem inferência |
| `remote-results.zip` | Resultados e estado para recuperação/retomada |

Uma execução `run` bem-sucedida produz as tabelas e o relatório. Ao usar somente
`generate`, haverá registros de geração, mas ainda não as métricas A–F.
Os gráficos `qexp_quality_cost.png` são produzidos pela célula de visualização
do notebook de expansão, não automaticamente pelo executor remoto.

Para recalcular as tabelas localmente a partir do download:

```bash
.venv/bin/python -m qexp.cli analyze \
  --config results/query_expansion/full/colab/pln-ID/configs/remote.json
```

Troque `pln-ID` pelo diretório real e `full` por `smoke` se necessário.
A análise usa a configuração salva com o resultado e não carrega LLM/BGE.

## 6. Retome ou recupere uma execução remota

### Retomar a partir de um ZIP já baixado

```bash
.venv/bin/python -m qexp.colab run \
  --config configs/full.json --gpu A100 \
  --restore results/query_expansion/full/colab/pln-ID/remote-results.zip
```

Uma nova sessão remota não recebe automaticamente o estado da anterior.
`--restore` envia os JSONs de gerações e checkpoints do ZIP indicado. As queries
concluídas só são reutilizadas se dados, parâmetros, código e versões forem
compatíveis; a query interrompida é refeita. Corpus, índices e pesos de modelos
não são incluídos nesse ZIP e podem precisar ser preparados novamente.

### Recuperar um download que falhou

Se a sessão ainda existir e o arquivo remoto já tiver sido criado:

```bash
.venv/bin/colab --auth oauth2 download -s NOME_DA_SESSAO \
  content/pln-ID-results.zip ./remote-results.zip
```

O executor imprime o nome da sessão e o caminho remoto. Com `--session`, o nome
da sessão pode ser diferente do identificador do arquivo; use ambos exatamente
como foram informados.

Depois de recuperar os resultados, encerre a sessão preservada:

```bash
.venv/bin/colab --auth oauth2 stop -s NOME_DA_SESSAO
```

Por padrão, uma sessão criada pelo executor é encerrada após o download validado.
`--keep` mantém essa sessão; `--session NOME` reutiliza uma existente e nunca a
encerra automaticamente. Uma falha de download também preserva a sessão.
Sessões preservadas podem continuar consumindo unidades de computação.
Se a VM for excluída antes de baixar os arquivos, a recuperação não é garantida.

Mais opções e detalhes estão no [guia específico do Colab](colab.md).

## 7. Execute a expansão localmente ou no notebook

### Terminal local

```bash
.venv/bin/python -m qexp.cli run --config configs/smoke.json
```

Esse comando roda **no seu computador**. Com `model_device: "auto"`, usa CUDA
local quando disponível e CPU caso contrário. Para usar a GPU remota, o módulo
é `qexp.colab`, como na seção 3.

Os resultados locais vão diretamente para `results/query_expansion/smoke/`.
Repetir o comando reaproveita os checkpoints compatíveis. Para outra variante,
crie um JSON com outro `output_dir` para preservar os resultados lado a lado.

Você também pode separar as etapas:

```bash
.venv/bin/python -m qexp.cli prepare --config configs/smoke.json
.venv/bin/python -m qexp.cli generate --config configs/smoke.json
.venv/bin/python -m qexp.cli run --config configs/smoke.json
.venv/bin/python -m qexp.cli analyze --config configs/smoke.json
```

`prepare` e `generate` são opcionais: `run` prepara o que estiver faltando.
Use `--progress plain` para mensagens de texto ou `--progress off` para
desativar o monitor. Para medir tempos novos, consulte as regras de cache no
[guia técnico](experiments.md#retomada-e-reutilização): `--no-resume` sozinho
não desativa o cache das gerações.

### Notebook de expansão

Localmente, inicie Jupyter na raiz do projeto:

```bash
.venv/bin/python -m jupyter lab
```

Abra `notebooks/query_expansion.ipynb`. Execute as células de localização do
projeto e instalação; depois configure:

```python
PROFILE = "smoke"  # "full" para o experimento completo
STAGE = "run"      # ou prepare, generate, analyze
RESUME = True
```

Execute a célula de configuração e a de execução. As células de gráficos e ZIP
podem ser usadas depois que existirem resultados de `run`.

No Colab pelo navegador, disponibilize a pasta do projeto no Drive, incluindo
`src/`, `configs/` e `pyproject.toml`. Em uma célula anterior à configuração:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Ajuste `PROJECT_ROOT` na célula do notebook para a pasta correta, por exemplo
`Path("/content/drive/MyDrive/PLN")`, e selecione um ambiente de execução com GPU.
Os notebooks novos importam o pacote: enviar somente o `.ipynb` não basta.
Esse fluxo pelo navegador é uma alternativa ao executor remoto do terminal.

## 8. Rode a comparação anterior BM25/BGE

1. Abra `scifact_bm25_reranker_colab.ipynb` no Colab pelo navegador e selecione GPU.
2. Execute a célula inicial de diretório de saída, a montagem do Drive,
   dependências e configurações, seguindo a ordem do notebook.
3. Mantenha `RUN_RERANKER = True` para comparar os dois pipelines.
4. Para uma verificação curta, use `MAX_QUERIES_PER_DATASET = 3` e desative
   `RUN_BM25_TOP_N_SWEEP`. Para avaliar todas as queries, use `None` no limite.
5. Execute o carregamento, métricas, inicialização do BGE e experimentos base.
   Depois execute as análises desejadas. Busca densa, sweep e visualizações 3D
   são análises adicionais; pule as células de 3D se não precisar delas.
6. Execute organização dos outputs e download do ZIP ao final.

A primeira célula imprime a pasta exclusiva `results/baseline/<data-hora>/`,
relativa ao diretório inicial do runtime. A organização final cria a subpasta
`results/` dentro dela. Guarde esse caminho ou extraia o ZIP baixado.
O notebook histórico não tem a retomada por query implementada para a expansão.

## 9. Rode o reranker seletivo anterior

Abra `notebooks/reranker_seletivo.ipynb` no Jupyter local ou no Colab. Execute
a localização do projeto e a instalação. Na célula de carregamento, configure:

```python
BASELINE_RESULTS = PROJECT_ROOT / "PLN_results_outputs"
```

Você pode apontar para a pasta de uma nova execução da comparação anterior ou
para um ZIP extraído. O leitor procura recursivamente:

- `rankings_all_datasets.csv`, contendo `doc_text`;
- `results_all_datasets.csv`, contendo as métricas e os tempos de BM25 e BGE.

Aponte para **uma única execução**. Depois execute, em ordem, o carregamento,
o classificador, o benchmark query-only e a auditoria da política. Esta última
depende dos resultados do benchmark. O BGE e os corpora não são carregados;
o benchmark pode baixar seu modelo de embeddings de queries.

Use múltiplos datasets e exemplos suficientes das duas classes para os folds.
A saída curta de três queries não é adequada para essa etapa; o benchmark da
MLP também exige amostras suficientes para sua validação interna.

Os arquivos ficam em `results/reranker_seletivo/<data-hora>/`, dentro do projeto.
A última célula cria o ZIP. No Colab, para baixá-lo:

```python
from google.colab import files
files.download(archive)
```

## 10. Problemas comuns

| Sintoma | O que verificar |
|---|---|
| `No module named qexp` ou `rich` | Refaça a instalação da seção 2 e use `.venv/bin/python`. |
| A100 recusada | Confira conta autenticada, saldo e disponibilidade. O executor não garante a alocação. |
| Comandos impressos, sem execução | Confira se usou `--dry-run`. |
| Demora antes da primeira query | Veja a etapa: pode estar instalando dependências, baixando modelos ou preparando BM25. |
| `CUDA out of memory` | Reduza `qexp.reranker_batch_size`; considere separar `generate` de `run` para liberar a memória do LLM entre processos. Veja como transferir o cache com `--restore` no guia Colab. |
| Mais de um CSV encontrado pelo seletivo | Aponte `BASELINE_RESULTS` para uma execução específica, não para a pasta de todas as execuções. |
| Erro de classes/folds no seletivo | Use os resultados completos, com múltiplos datasets e exemplos das duas classes. |
| Checkpoints não foram reutilizados | Confira `--restore` no remoto e a compatibilidade de código, parâmetros, versões e dados. |
| Falta `qexp_run.json` para `analyze` | Execute `run` primeiro ou aponte para os resultados baixados da sessão correta. |

## 11. Verificação offline

Para verificar o código com corpus pequeno e modelos simulados:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Os testes não alocam GPU nem usam créditos Colab. Eles não substituem a
execução completa com modelos reais. Para interpretar os resultados, consulte
[o protocolo e as hipóteses](query_expansion.md).
