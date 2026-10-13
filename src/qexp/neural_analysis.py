"""Artefatos adicionais da comparação neural, preservando a família anterior."""
import numpy as np
import pandas as pd

from .statistics import paired_statistics, holm, stable_seed


def export_neural_analysis(predictions, summary, models, neural_names, reference,
                           output, images, prefix, *, plots, seed, random_trials,
                           resamples, permutations):
    contrasts, random_rows = [], []
    references = list(dict.fromkeys([reference, "B", "uniform", "current"]))
    for dataset, g in predictions.groupby("dataset"):
        wide = g.pivot(index=["query_id", "group"], columns="policy", values="selected_ndcg")
        for name in neural_names:
            for control in references:
                stats = paired_statistics(wide[name]-wide[control], wide.index.get_level_values("group"),
                                          seed=stable_seed(seed, f"neural/{dataset}/{name}/{control}"),
                                          resamples=resamples, permutations=permutations)
                contrasts.append({"dataset": dataset, "policy": name, "reference": control, **stats})
            rows = g[g.policy == name]
            rng = np.random.default_rng(stable_seed(seed, f"neural/random/{dataset}/{name}"))
            records = rows.to_dict("records")
            for trial in range(random_trials):
                actions = rng.permutation(rows.selected_method.to_numpy())
                quality = [r["baseline_ndcg"] if a == "B" else r[f"ndcg_{a}"] for a, r in zip(actions, records)]
                costs = [r[f"seconds_{a}"] + r["estimated_policy_sec"]-r[f"seconds_{r['selected_method']}"]
                         for a, r in zip(actions, records)]
                random_rows.append({"dataset": dataset, "policy": name, "trial": trial,
                                    "ndcg": np.mean(quality), "estimated_sec": np.mean(costs)})
    contrasts = pd.DataFrame(contrasts)
    contrasts["p_holm"] = holm(contrasts.p_value)
    contrasts.to_csv(output/"neural_contrasts.csv", index=False)
    pd.DataFrame(random_rows).to_csv(output/"neural_random_matched.csv", index=False)
    metrics = ["selected_ndcg", "mean_delta", "realized_utility", "expansion_rate", "estimated_policy_sec"]
    stability = []
    for dataset, g in summary[summary.policy.isin(neural_names)].groupby("dataset"):
        for metric in metrics:
            stability.append({"dataset": dataset, "metric": metric, "seeds": len(g),
                              "mean": g[metric].mean(), "std": g[metric].std(ddof=1),
                              "min": g[metric].min(), "max": g[metric].max()})
    pd.DataFrame(stability).to_csv(output/"neural_seed_summary.csv", index=False)
    if not plots:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images.mkdir(parents=True, exist_ok=True)
    names = [prefix+"neural_effects", prefix+"neural_training"]
    if any((images/f"{name}.{ext}").exists() for name in names for ext in ("pdf", "png", "svg")):
        raise ValueError("Figuras neurais já existem; escolha outro nome de saída.")
    datasets = sorted(contrasts.dataset.unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(6*len(datasets), 4), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        rows = contrasts[(contrasts.dataset == dataset) & (contrasts.reference == reference)]
        ax.hlines(np.arange(len(rows)), rows.ci95_low, rows.ci95_high, color="#0072B2")
        ax.scatter(rows.mean_delta, np.arange(len(rows)), color="#0072B2")
        ax.axvline(0, color="black", lw=.7)
        ax.set_yticks(np.arange(len(rows)), [f"Semente {neural_names[n]}" for n in rows.policy])
        ax.set(title=dataset, xlabel=f"Delta nDCG neural − {reference}; IC95% marginal")
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(images/f"{names[0]}.{ext}", bbox_inches="tight", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(neural_names), figsize=(5*len(neural_names), 4), squeeze=False)
    for ax, name in zip(axes[0], neural_names):
        a = models[name].artifact
        for decay, h in pd.DataFrame(a["training_history"]).groupby("weight_decay"):
            line, = ax.plot(h.epoch, h.validation_loss, label=f"Validação; decay={decay:g}")
            ax.plot(h.epoch, h.train_loss, linestyle="--", color=line.get_color(), alpha=.65,
                    label=f"Treino; decay={decay:g}")
        ax.axvline(a["validation_choice"]["best_epoch"], color="black", lw=.7, label="Época escolhida")
        ax.set(title=f"Semente {neural_names[name]}", xlabel="Época", ylabel="SmoothL1 padronizada")
        ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(images/f"{names[1]}.{ext}", bbox_inches="tight", dpi=160)
    plt.close(fig)
    (images/f"{prefix}neural_gallery.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Seletor neural</title>' + ''.join(
            f'<h2>{n}</h2><a href="{n}.pdf">PDF</a><br><img style="max-width:100%" src="{n}.svg">' for n in names),
        encoding="utf-8")
