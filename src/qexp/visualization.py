"""Figuras reproduzíveis para o TCC: PNG para leitura, SVG/PDF para publicação."""
import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd


def render_study(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    summary = pd.read_csv(directory / "test_summary.csv")
    pipelines = pd.read_csv(directory / "pipeline_summary.csv")
    predictions = pd.read_csv(directory / "test_predictions.csv", dtype={"query_id": str})
    champions = pd.read_csv(directory / "validation_champions.csv").policy.tolist()
    thresholds = pd.read_csv(directory / "validation_thresholds.csv")
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180, "svg.fonttype": "none", "pdf.fonttype": 42})
    datasets = sorted(summary.dataset.unique())
    palette = ["#0072B2", "#D55E00", "#009E73"]
    saved = []

    def save(fig, name, caption):
        fig.text(.01, .005, "Exploratório · " + manifest["metric"] + " · consultas reservadas", fontsize=8, color="#555555")
        fig.tight_layout(rect=(0, .025, 1, .83 if fig.legends else .96))
        for suffix in ("png", "svg", "pdf"):
            fig.savefig(figures / f"{name}.{suffix}", bbox_inches="tight")
        plt.close(fig)
        saved.append((name, caption))

    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 5), squeeze=False, sharex=True)
    for ax, dataset in zip(axes[0], datasets):
        group = pipelines[(pipelines.dataset == dataset) & (pipelines.pipeline_id != "B")].sort_values("pipeline_id")
        y = np.arange(len(group))
        for position, row in zip(y, group.itertuples()):
            color = "#0072B2" if row.pipeline_id in {"G", "K"} else "#777777"
            ax.plot([row.ci95_low, row.ci95_high], [position, position], color=color)
            ax.scatter(row.mean_delta, position, color=color, marker="s" if row.p_holm < .05 else "o")
        ax.set(yticks=y, yticklabels=group.pipeline_id, title=dataset, xlabel="Δ nDCG frente a B (IC95%)")
        ax.axvline(0, color="black", lw=.8, ls="--")
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("Expansão uniforme: efeito pareado sobre o BGE")
    save(fig, "pipeline_effects", "IC por grupos. Azul: G/K com candidatos fixos. Quadrado: p ajustado por Holm < 0,05 nesta família.")

    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.5), squeeze=False, sharex=True)
    for ax, dataset in zip(axes[0], datasets):
        for i, (name, color) in enumerate(zip(champions, palette)):
            row = summary[(summary.dataset == dataset) & (summary.policy == name)].iloc[0]
            ax.plot([row.ci95_low, row.ci95_high], [i, i], color=color, lw=2)
            ax.scatter(row.mean_delta, i, color=color, marker="s" if row.p_holm < .05 else "o")
        ax.axvline(0, color="black", ls="--", lw=.8)
        ax.set(yticks=np.arange(len(champions)), yticklabels=champions, title=dataset,
               xlabel="Δ nDCG do seletor frente a B (IC95%)")
        ax.grid(axis="x", alpha=.2)
    save(fig, "selector_effects", "Efeito dos destaques congelados na validação. IC95% marginal por grupos; quadrado indica p Holm < 0,05 na família completa de seletores.")

    matrix = summary.pivot(index="policy", columns="dataset", values="mean_delta").sort_index()
    limit = max(float(np.abs(matrix.to_numpy()).max()), 1e-4)
    fig, ax = plt.subplots(figsize=(8, max(7, len(matrix) * .3)))
    image = ax.imshow(matrix, cmap="RdBu", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
    ax.set(yticks=np.arange(len(matrix)), yticklabels=matrix.index,
           xticks=np.arange(len(matrix.columns)), xticklabels=matrix.columns,
           title="Seletores × coleções: ganho médio frente a B")
    for i in range(len(matrix)):
        for j in range(len(matrix.columns)):
            value = matrix.iloc[i, j]
            ax.text(j, i, f"{value:+.4f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(value) > .6 * limit else "black")
    fig.colorbar(image, ax=ax, label="Δ nDCG")
    save(fig, "selector_heatmap", "Todas as versões predefinidas, nos mesmos pares. Cores indicam magnitude; não significância.")

    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.5), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        group = summary[summary.dataset == dataset]
        ax.scatter(group.estimated_policy_sec, group.selected_ndcg, color="#bbbbbb", alpha=.6, label="Outros seletores")
        base = group.iloc[0]
        ax.scatter(base.baseline_sec, base.baseline_ndcg, color="black", marker="X", s=80, label="B")
        for name, color in zip(champions, palette):
            row = group[group.policy == name].iloc[0]
            ax.scatter(row.estimated_policy_sec, row.selected_ndcg, s=65, color=color, label=name)
        ax.set(title=dataset, xlabel="Custo composto histórico (s/consulta)", ylabel="nDCG")
        ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", fontsize=8, ncol=2)
    save(fig, "quality_cost", "Destaques escolhidos na validação. Custo estimado por replay; não é nova medição de latência. Inclui geração, atributos e BGE inicial necessários.")

    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.5), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        group = summary[(summary.dataset == dataset) & summary.policy.isin(champions)].set_index("policy").loc[champions]
        y = np.arange(len(group))
        ax.barh(y, group.harm_rate, color="#D55E00", label="Piora")
        ax.barh(y, group.tie_rate, left=group.harm_rate, color="#dddddd", label="Empate")
        ax.barh(y, group.gain_rate, left=group.harm_rate + group.tie_rate, color="#0072B2", label="Melhora")
        ax.set(yticks=y, yticklabels=champions, xlim=(0, 1), title=dataset, xlabel="Proporção das consultas")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    save(fig, "wins_ties_losses", "Ganhos/empates/perdas em relação a B. A frequência é complementar à magnitude do efeito.")

    fig, axes = plt.subplots(len(champions), len(datasets), figsize=(5 * len(datasets), 3.3 * len(champions)), squeeze=False)
    for i, name in enumerate(champions):
        for j, dataset in enumerate(datasets):
            ax = axes[i, j]
            values = np.sort(predictions[(predictions.policy == name) & (predictions.dataset == dataset)].policy_delta)
            ax.bar(np.arange(len(values)), values, width=1, color=np.where(values < 0, "#D55E00", "#0072B2"))
            ax.axhline(0, color="black", lw=.5)
            ax.set(title=f"{dataset}\n{name}", xlabel="Consultas ordenadas pelo delta", ylabel="Δ nDCG")
    save(fig, "query_deltas", "Distribuição completa dos ganhos e perdas. Ordenar pelo delta é apenas visualização posterior, sem uso na decisão.")

    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.5), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        for i, (name, color) in enumerate(zip(champions, palette)):
            row = summary[(summary.dataset == dataset) & (summary.policy == name)].iloc[0]
            ax.plot([row.random_ndcg_p025, row.random_ndcg_p975], [i, i], lw=6, color="#cccccc")
            ax.scatter(row.random_ndcg_mean, i, color="black", marker="|", s=100)
            ax.scatter(row.selected_ndcg, i, color=color, s=60)
            ax.scatter(row.oracle_ndcg, i, color=color, marker="x", s=50)
        ax.set(yticks=np.arange(len(champions)), yticklabels=champions, title=dataset, xlabel="nDCG")
    save(fig, "random_oracle", "Ponto: seletor. Cinza: percentis 2,5–97,5 de 500 políticas aleatórias com mesmas quantidades de ações; traço: média aleatória. X: oráculo retrospectivo (não executável).")

    fig, axes = plt.subplots(1, len(champions), figsize=(5 * len(champions), 4.5), squeeze=False)
    for ax, name in zip(axes[0], champions):
        group = thresholds[thresholds.policy == name].copy()
        if "regularization" in group and group.regularization.notna().any():
            for reg, part in group.groupby("regularization"):
                ax.plot(part.expansion_rate, part.validation_gain, marker=".", ms=2, label=f"reg={reg:g}")
        else:
            ax.plot(group.expansion_rate, group.mean_delta_ndcg, marker=".", ms=2)
        ax.axhline(0, color="black", ls="--", lw=.8)
        ax.set(title=name, xlabel="Taxa de expansão na validação", ylabel="Ganho médio na validação")
        if group.regularization.notna().any():
            ax.legend(fontsize=8)
    save(fig, "validation_tradeoffs", "Curvas exclusivamente de validação. A logística original usa média por consulta; versões novas usam média das coleções. Não usar o teste para escolher o limiar.")

    fig, axes = plt.subplots(1, len(champions), figsize=(5 * len(champions), 4.5), squeeze=False)
    colors = ["#666666", "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#CC79A7"]
    for ax, name in zip(axes[0], champions):
        group = predictions[predictions.policy == name]
        counts = pd.crosstab(group.dataset, group.selected_method, normalize="index").reindex(columns=["B", "C", "D", "F", "G", "K"], fill_value=0)
        counts.plot.bar(stacked=True, ax=ax, color=colors, rot=0)
        ax.set(title=name, ylabel="Proporção das decisões", xlabel="", ylim=(0, 1))
        ax.legend(fontsize=8, ncol=3)
    save(fig, "action_mix", "Ações escolhidas por coleção. G/K preservam candidatos; C/D/F também mudam a recuperação. Falhas de geração podem resultar em B mesmo após solicitar expansão.")

    body = ["<!doctype html><html lang='pt-BR'><meta charset='utf-8'><title>Seletores de expansão — figuras</title>",
            "<style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 20px;color:#222}img{max-width:100%}figure{margin:40px 0}figcaption{line-height:1.5}a{color:#0072B2}</style>",
            "<h1>Expansão seletiva para melhorar o BGE</h1><p>Estudo exploratório. Destaques escolhidos na validação; gráficos das consultas reservadas. Consulte report.md e os CSVs para detalhes.</p>"]
    for name, caption in saved:
        body.append(f"<figure><h2>{html.escape(name.replace('_', ' '))}</h2><img src='{name}.png' alt='{html.escape(caption, quote=True)}'>"
                    f"<figcaption>{html.escape(caption)}<br><a href='{name}.pdf'>PDF</a> · <a href='{name}.svg'>SVG</a> · <a href='{name}.png'>PNG</a></figcaption></figure>")
    body.append("</html>")
    (figures / "gallery.html").write_text("\n".join(body), encoding="utf-8")
    return figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    print(render_study(args.study))


if __name__ == "__main__":
    main()
