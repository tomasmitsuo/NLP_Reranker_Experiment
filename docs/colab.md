# GPU remota da conta Colab

Para começar do zero, consulte o [guia passo a passo de execução](como_executar.md).

O [Google Colab CLI oficial](https://github.com/googlecolab/google-colab-cli)
permite autenticar sua conta, solicitar uma GPU e executar scripts remotamente.
Foi [anunciado pelo Google em junho de 2026](https://developers.googleblog.com/introducing-the-google-colab-cli/).
A GPU executa o processo Python no servidor: `torch.device("cuda")` no seu
computador continua apontando para uma GPU local, não para a GPU do Colab.

Esta integração usa a conta Colab e suas unidades de computação, sujeitas à
disponibilidade e aos recursos permitidos pelo plano. Não provisiona uma VM
separada de Compute Engine nem usa Colab Enterprise. A CLI oficial permite
consultar saldo e taxa de consumo com `colab usage`.

## Instalação e execução

Na raiz do projeto, em Linux ou macOS:

```bash
python -m pip install -e ".[colab]"
python -m qexp.colab run --config configs/smoke.json --gpu A100 --dry-run
python -m qexp.colab run --config configs/smoke.json --gpu A100
```

No ambiente virtual existente, substitua `python` por `.venv/bin/python`.
O modo `--dry-run` valida o bundle e mostra os comandos; não autentica,
não envia arquivos e não aloca GPU. A versão publicada 0.7.4 da CLI foi
inspecionada para conferir os argumentos utilizados pela integração.

Na primeira execução real, siga o fluxo de login impresso pela CLI oficial
(`--auth oauth2`): abra a URL, entre com a conta que tem a assinatura e forneça
o código no próprio terminal. O projeto não recebe senha nem armazena tokens;
a CLI oficial administra suas credenciais. Não é preciso criar uma chave de
API para o script. Consulte a [documentação de autenticação](https://github.com/googlecolab/google-colab-cli/blob/main/docs/04_automation_and_utility.md).

Para o experimento completo:

```bash
python -m qexp.colab run --config configs/full.json --gpu A100
```

O padrão do projeto é **A100**, conforme a preferência por uma GPU mais potente.
`--gpu` aceita T4, L4, A100, H100 e G4, conforme a CLI atual. A disponibilidade
depende da conta e da capacidade do serviço; Colab Pro não garante uma A100
em todas as sessões. Se a solicitação for recusada, o executor informa o erro
e não muda automaticamente para T4. `--high-mem` solicita mais RAM
quando a combinação de GPU/plano oferece essa opção. O script exige CUDA na VM
e informa erro se não houver GPU; não troca silenciosamente para CPU.

## O que o executor faz

1. Valida a configuração e empacota `src/`, `pyproject.toml` e o protocolo escolhido.
2. Cria uma sessão GPU na conta autenticada (ou conecta a uma sessão informada).
3. Envia o pacote, instala suas dependências na VM e executa `qexp.cli` remotamente.
4. Baixa um ZIP com resultados, log, configuração, gerações e checkpoints.
5. Confere o status do experimento e a integridade do ZIP antes de encerrar uma
   sessão criada pelo próprio comando.

Datasets, modelos, `.env`, credenciais, `.git`, resultados anteriores e o
ambiente virtual local não entram no bundle do código. As bases e modelos são
baixados diretamente na VM. Um snapshot anterior só é enviado com `--restore`.

A integração automatizada cobre o experimento de expansão (`run`/`generate`).
Os notebooks anteriores continuam disponíveis no Colab pelo navegador ou pela
[extensão oficial para VS Code](https://github.com/googlecolab/colab-vscode),
que também suporta runtimes da conta Colab. Não é necessário rerodar o BGE para
o notebook de reranker seletivo, pois ele usa os CSVs existentes.

## Resultados e retomada

O caminho local é impresso no terminal:

```text
<output_dir da configuração>/colab/pln-<id>/
  remote-results.zip
  remote_status.json
  experiment.log
  configs/remote.json
  outputs/qexp_*.csv, qexp_run.json, ...
  state/generations/*.json
  state/checkpoints/<dataset>/*.json
```

Para retomar queries concluídas em uma nova VM:

```bash
python -m qexp.colab run --config configs/full.json --gpu A100 \
  --restore results/query_expansion/full/colab/pln-ID/remote-results.zip
```

São restaurados apenas JSONs de gerações/checkpoints. Índices, datasets e pesos
de modelos não são transferidos no snapshot. Os checkpoints só são reutilizados
quando código, versões, dados e parâmetros correspondem aos da execução original.
Use revisões fixas de modelos para o experimento final. Custos salvos continuam
sendo históricos, inclusive após mudar de GPU.

Para preencher apenas o cache do LLM, use `generate` em vez de `run` e passe o
ZIP produzido com `--restore` à execução seguinte. Para refazer tabelas
localmente após o download, sem GPU:

```bash
python -m qexp.cli analyze \
  --config results/query_expansion/full/colab/pln-ID/configs/remote.json
```

## Sessões e falhas

- `--keep` preserva a sessão criada mesmo após baixar os resultados.
- `--session NOME` usa uma sessão existente e nunca a encerra automaticamente;
  `--gpu`/`--high-mem` só selecionam hardware ao criar uma sessão nova. Cada envio
  recebe uma pasta isolada; use `--restore` para reutilizar estado de outra execução.
- `--timeout SEGUNDOS` configura o timeout da execução remota (padrão: 43200).
  Isso não prolonga os limites do plano nem garante permanência da VM.
- Se o experimento falhar, o worker tenta salvar log e checkpoints concluídos
  no ZIP. O comando retorna erro, mesmo que a CLI não propague o erro do kernel.
- Se o download falhar ou a conexão for interrompida, a sessão é preservada e
  seu nome/caminho de recuperação são impressos. Ela pode continuar consumindo
  unidades de computação. Após a execução terminar, recupere o ZIP e encerre:

```bash
colab --auth oauth2 download -s NOME content/pln-ID-results.zip ./remote-results.zip
colab --auth oauth2 stop -s NOME
```

A recuperação depende de a VM ainda existir. Os checkpoints ficam no disco da
VM até serem baixados; se o Colab encerrar a VM antes disso, não há garantia de
recuperá-los. O modo remoto não monta o Drive automaticamente.

A validação local usa testes simulados para autenticação/comandos e não aloca
recursos. Uma execução real depende do login Google e de uma GPU disponível
na conta do usuário.

## Progresso durante a execução

O executor transmite as etapas e o progresso global de queries ao terminal.
Mensagens distinguem queries novas e checkpoints reutilizados, e identificam
dataset, query e pipeline atual. Há também avisos de atividade a cada 30 segundos
durante etapas longas, incluindo instalação de dependências.

O estado detalhado é salvo em `outputs/qexp_progress.json` na VM e incluído no
ZIP baixado, junto com `experiment.log`. Veja o [guia de progresso](experiments.md#acompanhamento-do-progresso).
Não é necessário acrescentar uma opção ao comando de execução remota.
