"""Estudo delimitado do seletor: candidatos fixos, ablações e políticas congeladas.

python -m qexp.final_study --run SESSAO/outputs/qexp_run.json --output SESSAO/selector
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .benefit_selection import BenefitSelector, fit_benefit_selector, macro_mean
from .checkpoints import load_result
from .config import CONTROLLED_METHODS, FIXED_METHODS, DATASETS
from .learned_selection import make_splits
from .selector_study import make_domain_splits
from .selector_features import uses_bge, uses_rarity, RARITY_FEATURES, FEATURE_SETS
from .statistics import paired_statistics, holm, stable_seed
from .utility_selection import UtilitySelector, fit_utility_selector, utility_table


def verify_fixed_candidates(result, methods):
    """Compara IDs, não apenas Candidate Recall, e a identidade da expansão."""
    rankings = result["rankings"]
    candidates = rankings[rankings.stage == "candidates"]
    lists = {key: tuple(g.sort_values("rank").doc_id) for key, g in
             candidates.groupby(["dataset", "query_id", "pipeline_id"])}
    artifacts = {(a["dataset"], a["query_id"], a["pipeline_id"]): a for a in result["artifacts"]}
    base = result["metrics"].query("pipeline_id == 'B'")
    for row in base.itertuples():
        key = (row.dataset, row.query_id)
        original = lists.get((*key, "B"))
        if not original or len(set(original)) != len(original):
            raise ValueError("Snapshot sem candidatos B únicos.")
        expansions = set()
        for method in methods:
            if lists.get((*key, method)) != original:
                raise ValueError(f"Candidatos diferentes de B: {key}/{method}")
            if method in CONTROLLED_METHODS:
                a = artifacts[(*key, method)]
                expansions.add((a["cache_key"], a["controlled_expansion"]))
        if len(expansions) > 1:
            raise ValueError("Os mecanismos controlados não usaram a mesma expansão.")


def select_alternatives(frame, methods, limit, harm_penalty, cost_weight):
    """Escolha gulosa por utilidade complementar somente na validação."""
    val = frame[frame.split == "validation"]
    utility = {m: val[f"delta_{m}"].to_numpy() - harm_penalty*np.maximum(-val[f"delta_{m}"], 0)
               - cost_weight*val[f"extra_sec_{m}"] for m in methods}
    selected, best = [], np.zeros(len(val))
    while len(selected) < min(limit, len(methods)):
        method = max(sorted(set(methods)-set(selected)),
                     key=lambda m: macro_mean(np.maximum(best, utility[m]), val.dataset))
        selected.append(method)
        best = np.maximum(best, utility[method])
    return sorted(selected)


def predictions_for(result, frame, model, name, methods, uniform):
    test = frame[frame.split == "test"]
    artifacts = {(a["dataset"], a["query_id"]): a for a in result["artifacts"] if a["pipeline_id"] == "B"}
    rows = []
    for r in test.to_dict("records"):
        if model:
            model.check_unseen(r["dataset"], r["query_id"], r["query_text"])
            action, score = model.choose(r)
        else:
            action, score = name, 0.
        extra_features = artifacts[(r["dataset"], r["query_id"])]
        overhead = (extra_features.get("feature_lexical_sec", 0) +
                    (extra_features.get("feature_bge_sec", 0) if uses_bge(model.artifact["feature_set"]) else 0)) if model else 0
        if model and uses_rarity(model.artifact["feature_set"]):
            overhead += extra_features.get("feature_corpus_sec", 0)
        chosen = r["baseline_ndcg"] if action == "B" else r[f"ndcg_{action}"]
        row = {**r, "policy": name, "selected_method": action, "decision_score": score,
               "selected_ndcg": chosen, "policy_delta": chosen-r["baseline_ndcg"],
               "expand": action != "B", "estimated_policy_sec": r[f"seconds_{action}"] + overhead,
               "uniform_ndcg": r["baseline_ndcg"] if uniform == "B" else r[f"ndcg_{uniform}"],
               "oracle_ndcg": max(r["baseline_ndcg"], *[r[f"ndcg_{m}"] for m in methods])}
        if isinstance(model, UtilitySelector):
            for method, values in model.components(r).items():
                row.update({f"predicted_{component}_{method}": v for component, v in values.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def export_alternative_comparisons(frame, methods, output, *, seed=42, resamples=2000, permutations=9999):
    alternatives, incorporation = [], []
    for dataset, g in frame[frame.split == "test"].groupby("dataset"):
        values = {"B": g.baseline_ndcg, **{m: g[f"ndcg_{m}"] for m in methods}}
        for method in ["B", *methods]:
            alternatives.append({"dataset": dataset, "method": method, "queries": len(g),
                                 "ndcg": values[method].mean(), "estimated_sec": g[f"seconds_{method}"].mean()})
        for i, method in enumerate(methods):
            for reference in ["B", *methods[:i]]:
                stats = paired_statistics(values[method]-values[reference], g.group,
                                          seed=stable_seed(seed, f"incorporation/{dataset}/{method}/{reference}"),
                                          resamples=resamples, permutations=permutations)
                incorporation.append({"dataset": dataset, "method": method, "reference": reference, **stats})
    incorporation = pd.DataFrame(incorporation)
    incorporation["p_holm"] = holm(incorporation.p_value)
    incorporation.to_csv(output/"incorporation_comparisons.csv", index=False)
    pd.DataFrame(alternatives).to_csv(output/"alternatives_test.csv", index=False)


def export_query_buckets(frame, methods, output):
    """Diagnóstico descritivo; limites são quantis de treino, sem otimização pelo teste."""
    rows, limits = [], {}
    attributes = ["query_words"] + (["query_idf_mean", "query_rare_ratio", "query_oov_ratio"]
                                    if set(RARITY_FEATURES) <= set(frame) else [])
    train = frame[frame.split == "train"]
    test = frame[frame.split == "test"].copy()
    for attribute in attributes:
        cuts = sorted(set(float(v) for v in train[attribute].quantile([1/3, 2/3])))
        limits[attribute] = cuts
        test["bucket"] = pd.cut(test[attribute], [-np.inf, *cuts, np.inf]).astype(str)
        for (dataset, bucket), group in test.groupby(["dataset", "bucket"]):
            for method in methods:
                delta = group[f"delta_{method}"]
                rows.append({"attribute": attribute, "dataset": dataset, "bucket": bucket,
                             "method": method, "queries": len(group), "mean_delta": delta.mean(),
                             "gain_rate": float((delta > 1e-12).mean()),
                             "harm_rate": float((delta < -1e-12).mean()),
                             "mean_harm": np.maximum(-delta, 0).mean()})
    pd.DataFrame(rows).to_csv(output/"query_buckets.csv", index=False)
    (output/"query_bucket_limits.json").write_text(json.dumps({"partition": "train", "cuts": limits}, indent=2))


def render(summary, predictions, images, metric, prefix="", overwrite=False):
    if not overwrite and images.exists() and any(images.glob(f"{prefix}*.pdf")):
        raise ValueError("Figuras deste estudo já existem; escolha outro nome de saída.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    images.mkdir(parents=True, exist_ok=True)
    datasets = sorted(summary.dataset.unique())
    names = []
    labels = {"B": "B: original", "uniform": "Expansão uniforme", "current": "Seletor por ganho",
              "proposed": "Proposta", "no_bge_signals": "Sem sinais BGE", "no_harm_penalty": "Sem penalizar perdas",
              "no_cost_penalty": "Sem penalizar custo", "cost_x2": "Peso de custo dobrado", "length_only": "Só tamanho",
              "rarity_only": "Só raridade", "length_rarity": "Tamanho + raridade", "bge_rarity": "Sinais BGE + raridade"}
    policies = sorted(summary.policy.unique())
    colors = plt.get_cmap("tab20")
    markers = ['o', 's', '^', 'v', 'D', 'P', 'X', '*']
    for kind in ("quality_cost", "query_deltas", "ablation_effects"):
        fig, axes = plt.subplots(1, len(datasets), figsize=(6*len(datasets), 4.8), squeeze=False)
        for ax, dataset in zip(axes[0], datasets):
            s = summary[summary.dataset == dataset]
            if kind == "quality_cost":
                for r in s.itertuples():
                    i = policies.index(r.policy)
                    ax.scatter(r.estimated_policy_sec, r.selected_ndcg, color=colors(i),
                               marker=markers[i % len(markers)], s=55, alpha=.75,
                               label=labels.get(r.policy, r.policy))
                ax.set(xlabel="Custo histórico composto (s/consulta)", ylabel=metric)
            elif kind == "query_deltas":
                p = predictions[(predictions.dataset == dataset) & (predictions.policy == "proposed")]
                values = np.sort(p.policy_delta)
                ax.bar(np.arange(len(values)), values, color=["#D55E00" if v < 0 else "#0072B2" for v in values])
                ax.axhline(0, color="black", lw=.6)
                if not p.expand.any():
                    ax.text(.5, .55, "Nenhuma expansão selecionada", transform=ax.transAxes, ha="center", fontsize=10)
                ax.set(xlabel="Consultas ordenadas pelo delta", ylabel=f"Delta {metric} proposto − B")
            else:
                s = s[~s.policy.isin(["B", "uniform"])].reset_index(drop=True)
                # Os ICs percentis não precisam conter a estimativa pontual.
                ax.hlines(np.arange(len(s)), s.ci95_low, s.ci95_high, color="#0072B2")
                ax.scatter(s.mean_delta, np.arange(len(s)), color="#0072B2")
                ax.set_yticks(np.arange(len(s)), [labels.get(p, p) for p in s.policy])
                ax.axvline(0, color="black", lw=.6)
                ax.set(xlabel=f"Delta {metric} − B; IC95% marginal")
            ax.set_title(dataset)
        if kind == "quality_cost":
            handles, texts = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, texts, loc="lower center", ncol=4, fontsize=9)
            fig.tight_layout(rect=(0, .22, 1, 1))
        else:
            fig.tight_layout()
        for ext in ("png", "pdf", "svg"):
            fig.savefig(images/f"{prefix}{kind}.{ext}", bbox_inches="tight", dpi=160)
        plt.close(fig)
        names.append(prefix+kind)
    (images/f"{prefix}gallery.html").write_text('<!doctype html><meta charset="utf-8"><title>Estudo do seletor</title>' +
        ''.join(f'<h2>{n}</h2><a href="{n}.pdf">PDF</a><br><img style="max-width:100%" src="{n}.svg">' for n in names))


def run_final_study(run, output, *, methods=("G", "K"), splits=None, test_dataset=None,
                    max_methods=2, harm_penalty=1., cost_weight=.001, seed=42,
                    resamples=2000, permutations=9999, random_trials=500, plots=True, images_dir=None, query_features="none",
                    neural=False, neural_seeds=(42, 43, 44), neural_feature_set="bge",
                    neural_max_epochs=300, neural_patience=30):
    run, output = Path(run).resolve(), Path(output).resolve()
    from .naming import internal_id
    methods = sorted(set(map(internal_id, methods)))
    if query_features not in {"none", "length", "all"}:
        raise ValueError("query_features deve ser none, length ou all.")
    if neural and (not neural_seeds or len(set(neural_seeds)) != len(neural_seeds)
                   or any(type(s) is not int or not 0 <= s < 2**32 for s in neural_seeds)
                   or neural_feature_set not in FEATURE_SETS
                   or any(type(v) is not int or v < 1 for v in (neural_max_epochs, neural_patience))):
        raise ValueError("Sementes, atributos ou limites neurais inválidos.")
    if (not methods or not set(methods) <= FIXED_METHODS or not 1 <= max_methods <= len(methods)
            or (splits and test_dataset) or random_trials < 1
            or not np.isfinite([harm_penalty, cost_weight]).all() or min(harm_penalty, cost_weight) < 0):
        raise ValueError("Métodos, partições ou parâmetros inválidos.")
    if output.exists():
        raise ValueError("Use uma pasta nova para preservar os resultados anteriores.")
    result = load_result(run)
    verify_fixed_candidates(result, methods)
    partition = (pd.read_csv(splits, dtype=str, keep_default_na=False) if splits else
                 make_domain_splits(result, test_dataset, seed) if test_dataset else make_splits(result, seed))
    output.mkdir(parents=True)
    split_path = output/"splits.csv"
    partition.to_csv(split_path, index=False)
    frame = utility_table(result, methods, split_path)
    if query_features == "all" and not set(RARITY_FEATURES) <= set(frame):
        raise ValueError("Snapshot sem raridade; use coleta com collect_query_statistics=true ou --query-features length.")
    if neural and not set(FEATURE_SETS[neural_feature_set]) <= set(frame):
        raise ValueError("Snapshot sem atributos neurais; raridade exige collect_query_statistics=true na coleta.")
    selected = select_alternatives(frame, methods, max_methods, harm_penalty, cost_weight)
    val = frame[frame.split == "validation"]
    # Melhor alternativa aplicada sempre; B permanece como controle separado.
    uniform_scores = {"B": 0., **{m: macro_mean(val[f"delta_{m}"] - harm_penalty*np.maximum(-val[f"delta_{m}"], 0)
                          - cost_weight*val[f"extra_sec_{m}"], val.dataset) for m in methods}}
    uniform = max(methods, key=uniform_scores.get)
    session_images = run.parent.parent/"images" if run.parent.name == "outputs" and output.parent == run.parent.parent else output/"images"
    images = Path(images_dir).resolve() if images_dir else session_images
    prefix = output.name+"_" if images != output/"images" else ""
    gallery = os.path.relpath(images/f"{prefix}gallery.html", output)
    manifest = {"images": str(images), "status": "running", "source": str(run), "source_sha256": hashlib.sha256(run.read_bytes()).hexdigest(),
                "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))},
                "query_features": query_features, "requested_methods": methods, "selected_methods": selected, "uniform": uniform,
                "uniform_validation_utilities": uniform_scores,
                "harm_penalty": harm_penalty, "cost_weight": cost_weight, "seed": seed,
                "resamples": resamples, "permutations": permutations, "random_trials": random_trials,
                "metric": f"ndcg@{result['metadata']['config']['eval_k']}",
                "scope": "Exploratório; reparticionar dados já analisados não produz confirmação independente.",
                "cost": "Replay de custos compostos históricos; inclui atributos, exclui nova latência do seletor. Pesos expressos em nDCG por segundo; não impõem teto de tempo.",
                "statistics": "Contrastes predefinidos em test_contrasts.csv: proposto vs B/uniforme/atual e ablações vs proposto; Holm conjunto, ICs marginais. Controles aleatórios não são testes de equivalência."}
    manifest_path = output/"manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    variants = {"proposed": ("bge", harm_penalty, cost_weight),
                "no_bge_signals": ("lexical", harm_penalty, cost_weight),
                "no_harm_penalty": ("bge", 0., cost_weight),
                "no_cost_penalty": ("bge", harm_penalty, 0.)}
    if query_features in {"length", "all"}:
        variants["length_only"] = ("length", harm_penalty, cost_weight)
    if query_features == "all":
        variants.update({"rarity_only": ("rarity", harm_penalty, cost_weight),
                         "length_rarity": ("length_rarity", harm_penalty, cost_weight),
                         "bge_rarity": ("bge_rarity", harm_penalty, cost_weight)})
    # Pontos da curva de custo são predefinidos, não escolhidos pelo teste.
    if cost_weight > 0:
        variants["cost_x2"] = ("bge", harm_penalty, 2*cost_weight)
    neural_names = {f"neural_seed_{s}": s for s in neural_seeds} if neural else {}
    neural_reference = None
    if neural:
        from .neural_selection import NeuralUtilitySelector, fit_neural_selector
        matched = (neural_feature_set, harm_penalty, cost_weight)
        neural_reference = next((name for name, params in variants.items() if params == matched), "neural_reference")
        if neural_reference not in variants:
            variants[neural_reference] = matched
        manifest["neural"] = {"seeds": list(neural_seeds), "feature_set": neural_feature_set,
                              "reference": neural_reference, "hidden_units": 16,
                              "max_epochs": neural_max_epochs, "patience": neural_patience,
                              "statistics": "Família adicional em neural_contrasts.csv: cada semente vs Ridge com mesmos atributos, B, uniforme e atual; Holm conjunto. Sementes não são consultas independentes."}
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    models = {}
    for name in ["current", *variants, *neural_names]:
        print(f"Ajustando {name}: treino/validação", flush=True)
        if name == "current":
            artifact, thresholds = fit_benefit_selector(result, split_path, methods=selected, feature_set="bge", frame=frame)
            model = BenefitSelector(artifact)
        elif name in neural_names:
            artifact, thresholds = fit_neural_selector(result, split_path, selected, neural_feature_set,
                                                       harm_penalty, cost_weight, seed=neural_names[name],
                                                       max_epochs=neural_max_epochs, patience=neural_patience, frame=frame)
            model = NeuralUtilitySelector(artifact)
        else:
            features, penalty, cost = variants[name]
            artifact, thresholds = fit_utility_selector(result, split_path, selected, features, penalty, cost, frame=frame)
            model = UtilitySelector(artifact)
        artifact.update(policy_name=name, source_sha256=manifest["source_sha256"])
        model.artifact.update(policy_name=name, source_sha256=manifest["source_sha256"])
        models[name] = model
        folder = output/"models"/name
        folder.mkdir(parents=True)
        (folder/"selector.json").write_text(json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False))
        thresholds.to_csv(folder/"validation_thresholds.csv", index=False)
        if name in neural_names:
            pd.DataFrame(artifact["training_history"]).to_csv(folder/"training_history.csv", index=False)
        datasets = sorted(set(frame.dataset))
        profile = {"project_root": str(Path(__file__).resolve().parents[2]),
                   "datasets": [d["slug"] for d in DATASETS if d["name"] in datasets],
                   "query_split_file": str(split_path), "query_split": "test",
                   "output_dir": str(output/"live"/name), "checkpoint_dir": str(output/"live_checkpoints"/name),
                   "qexp": {**result["metadata"]["config"], "pipelines": ["L"], "selective_method": artifact["method"],
                            "selector_model_path": str(folder/"selector.json"), "cache_dir": str(output/"live_cache"/name),
                            "max_queries_per_dataset": None}}
        (folder/"run_config.json").write_text(json.dumps(profile, indent=2, ensure_ascii=False))
    # Todas as escolhas acima estão congeladas antes de avaliar as consultas de teste.
    prediction_frames = [predictions_for(result, frame, model, name, methods, uniform) for name, model in models.items()]
    prediction_frames += [predictions_for(result, frame, None, "B", methods, uniform),
                          predictions_for(result, frame, None, uniform, methods, uniform).assign(policy="uniform")]
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(output/"test_predictions.csv", index=False)
    summaries, contrasts, random_rows = [], [], []
    for (name, dataset), g in predictions.groupby(["policy", "dataset"]):
        stats = paired_statistics(g.policy_delta, g.group, seed=stable_seed(seed, f"{name}/{dataset}"),
                                  resamples=resamples, permutations=permutations)
        extra = np.maximum(g.estimated_policy_sec-g.seconds_B, 0)
        summaries.append({"policy": name, "dataset": dataset, **stats,
                          "selected_ndcg": g.selected_ndcg.mean(), "baseline_ndcg": g.baseline_ndcg.mean(),
                          "estimated_policy_sec": g.estimated_policy_sec.mean(), "expansion_rate": g.expand.mean(),
                          "oracle_ndcg": g.oracle_ndcg.mean(), "oracle_regret": (g.oracle_ndcg-g.selected_ndcg).mean(),
                          "realized_utility": (g.policy_delta-harm_penalty*np.maximum(-g.policy_delta, 0)-cost_weight*extra).mean()})
        if name == "proposed":
            rng = np.random.default_rng(stable_seed(seed, dataset))
            for trial in range(random_trials):
                actions = rng.permutation(g.selected_method.to_numpy())
                quality = [r["baseline_ndcg"] if a == "B" else r[f"ndcg_{a}"] for a, r in zip(actions, g.to_dict('records'))]
                # Mesmo overhead do seletor; tempo da ação muda conforme a consulta sorteada.
                cost = [r[f"seconds_{a}"] + (r["estimated_policy_sec"]-r[f"seconds_{r['selected_method']}"])
                        for a, r in zip(actions, g.to_dict('records'))]
                random_rows.append({"dataset": dataset, "trial": trial, "ndcg": np.mean(quality), "estimated_sec": np.mean(cost)})
    summary = pd.DataFrame(summaries)
    pairs = [("proposed", r) for r in ("B", "uniform", "current")] + [(n, "proposed") for n in variants if n not in {"proposed", "neural_reference"}]
    for dataset, g in predictions.groupby("dataset"):
        wide = g.pivot(index=["query_id", "group"], columns="policy", values="selected_ndcg")
        for left, right in pairs:
            stats = paired_statistics(wide[left]-wide[right], wide.index.get_level_values("group"),
                                      seed=stable_seed(seed, f"{dataset}/{left}/{right}"), resamples=resamples, permutations=permutations)
            contrasts.append({"dataset": dataset, "policy": left, "reference": right, **stats})
    contrasts = pd.DataFrame(contrasts)
    contrasts["p_holm"] = holm(contrasts.p_value)
    random = pd.DataFrame(random_rows)
    random.to_csv(output/"random_matched.csv", index=False)
    for dataset, g in random.groupby('dataset'):
        mask = (summary.policy == 'proposed') & (summary.dataset == dataset)
        observed = summary.loc[mask, 'selected_ndcg'].iloc[0]
        summary.loc[mask, 'random_ndcg_mean'] = g.ndcg.mean()
        summary.loc[mask, 'random_ndcg_p025'] = g.ndcg.quantile(.025)
        summary.loc[mask, 'random_ndcg_p975'] = g.ndcg.quantile(.975)
        summary.loc[mask, 'random_percentile'] = float(((g.ndcg < observed-1e-12).astype(float) + .5*np.isclose(g.ndcg, observed, rtol=0, atol=1e-12)).mean())
        summary.loc[mask, 'random_estimated_sec_mean'] = g.estimated_sec.mean()
    summary.to_csv(output/"test_summary.csv", index=False)
    contrasts.to_csv(output/"test_contrasts.csv", index=False)
    export_alternative_comparisons(frame, methods, output, seed=seed, resamples=resamples, permutations=permutations)
    if query_features != "none":
        export_query_buckets(frame, methods, output)
    from .intent_audit import export_intent_sample, audit_development_generations
    export_intent_sample(result, frame, methods, output/"intent", seed=seed)
    audit_development_generations(result, frame).to_csv(output/"generation_audit_development.csv", index=False)
    if plots:
        render(summary, predictions, images, manifest["metric"], prefix)
    if neural:
        from .neural_analysis import export_neural_analysis
        export_neural_analysis(predictions, summary, models, neural_names, neural_reference,
                               output, images, prefix, plots=plots, seed=seed, random_trials=random_trials,
                               resamples=resamples, permutations=permutations)
    lines = ["# Estudo do seletor por utilidade", "", manifest["scope"], "", f"Alternativas selecionadas na validação: {', '.join(selected)}. Controle uniforme: {uniform}.",
             "", "A seleção de alternativas é mantida nas ablações: elas isolam os componentes do seletor, não refazem toda a busca de métodos.",
             "", manifest["cost"], "", manifest["statistics"], "", summary.to_markdown(index=False),
             "", "## Contrastes pareados", "", contrasts.to_markdown(index=False),
             "", "## Alternativas uniformes", "", "alternatives_test.csv e incorporation_comparisons.csv comparam todas as alternativas nas mesmas consultas de teste; Holm em família secundária separada. Não selecionar um vencedor pelo teste.",
             "", "## Tamanho e raridade", "", f"Grupos adicionais de atributos: {query_features}. Quando habilitados, query_buckets.csv resume ganhos por faixas delimitadas exclusivamente no treino; diagnóstico descritivo, sem afirmar causalidade. Raridade ausente nunca é preenchida com valores inventados.",
             "", "## Controle aleatório", "", "random_matched.csv preserva as contagens de ações por coleção. Não garante mesmo tempo: custos são exportados por sorteio.",
             "", "## Preservação da intenção", "", "Preencher intent/annotations.csv; rótulos humanos permanecem vazios. Não há evidência semântica até essa avaliação.",
             "", "## Execução congelada", "", "models/proposed/run_config.json executa somente a política e A/B. Usar sessão nova, sem geração antecipada, para observar policy_wall_sec; identificar setup, cache e checkpoints.",
             "", f"[Gráficos]({gallery})"]
    (output/"report.md").write_text('\n'.join(lines)+'\n', encoding='utf-8')
    if neural:
        with (output/"report.md").open("a", encoding="utf-8") as report:
            report.write("\n## Extensão neural\n\n" + manifest["neural"]["statistics"] +
                         f"\n\nReferência controlada: {neural_reference}. Atributos: {neural_feature_set}. "
                         "Todas as sementes são reportadas, sem escolher pelo teste. "
                         "neural_seed_summary.csv resume variabilidade entre sementes; não é IC de generalização. "
                         "neural_random_matched.csv preserva as contagens de ações de cada semente por coleção. "
                         "O replay exclui nova latência da decisão; os perfis models/neural_seed_*/run_config.json "
                         "permitem medir a política congelada em sessão nova.\n")
            if plots:
                report.write(f"\n[Gráficos neurais]({os.path.relpath(images/f'{prefix}neural_gallery.html', output)})\n")
    manifest.update(status="completed", partition_counts=frame.groupby(["dataset", "split"]).size().rename("queries").reset_index().to_dict('records'))
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--methods', nargs='+', default=['G', 'K'])
    p.add_argument('--splits', type=Path)
    p.add_argument('--test-dataset')
    p.add_argument('--max-methods', type=int, default=2)
    p.add_argument('--harm-penalty', type=float, default=1.)
    p.add_argument('--cost-weight', type=float, default=.001)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resamples', type=int, default=2000)
    p.add_argument('--permutations', type=int, default=9999)
    p.add_argument('--random-trials', type=int, default=500)
    p.add_argument('--query-features', choices=['none', 'length', 'all'], default='none')
    p.add_argument('--neural', action='store_true', help='Acrescenta MLP e comparação controlada com Ridge, mantendo as políticas existentes.')
    p.add_argument('--neural-seeds', type=int, nargs='+', default=[42, 43, 44])
    p.add_argument('--neural-feature-set', choices=sorted(FEATURE_SETS), default='bge')
    p.add_argument('--neural-max-epochs', type=int, default=300)
    p.add_argument('--neural-patience', type=int, default=30)
    p.add_argument('--images-dir', type=Path)
    p.add_argument('--no-plots', action='store_true')
    args = vars(p.parse_args())
    args['plots'] = not args.pop('no_plots')
    print(run_final_study(**args))


if __name__ == '__main__':
    main()
