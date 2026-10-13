"""Cruzamento completo técnica × seletor, com partições comuns e perfis executáveis.

python -m qexp.matrix_study --run SESSAO/outputs/qexp_run.json --output SESSAO/matrix
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .benefit_selection import BenefitSelector, fit_benefit_selector
from .checkpoints import load_result
from .config import ALL_TECHNIQUES, DATASETS, FIXED_METHODS, QEXP_CONFIG, QEXP_PIPELINES
from .final_study import verify_fixed_candidates
from .learned_selection import LearnedSelector, fit_selector, make_splits
from .naming import internal_id, public_id, public_config, public_model
from .neural_selection import NeuralUtilitySelector, fit_neural_selector
from .selection import qexp_should_expand
from .selector_features import FEATURE_SETS, uses_bge, uses_rarity
from .statistics import paired_statistics, holm, stable_seed
from .utility_selection import UtilitySelector, fit_utility_selector, utility_table

TECHNIQUES = ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "Q_CONCAT", "Q_REPLACE", "Q_SCORE", "Q_RRF"]
ESTIMATORS = ["logistic", "weighted_logistic", "gain", "utility", "neural"]


def validate_options(methods=TECHNIQUES, selectors=("s1", "s2", "s3", "s4"),
                     estimators=ESTIMATORS, neural_seeds=(42, 43, 44), feature_set="bge", **unused):
    if (not methods or len(set(map(internal_id, methods))) != len(methods)
            or not set(map(internal_id, methods)) <= ALL_TECHNIQUES
            or not selectors or len(set(selectors)) != len(selectors) or not set(selectors) <= {"s1", "s2", "s3", "s4"}
            or not estimators or len(set(estimators)) != len(estimators) or not set(estimators) <= set(ESTIMATORS)
            or not neural_seeds or len(set(neural_seeds)) != len(neural_seeds)
            or any(type(s) is not int or not 0 <= s < 2**32 for s in neural_seeds)
            or feature_set not in FEATURE_SETS
            or ("s3" in selectors and "logistic" in estimators and feature_set not in {"lexical", "bge"})):
        raise ValueError("Matriz inválida. Logística histórica aceita lexical/bge; demais modelos aceitam todos os grupos.")


def _profile(result, split_path, folder, method, selector, model_path=None):
    config = {**QEXP_CONFIG, **result["metadata"]["config"], "id_scheme": "technique_selector",
              "pipelines": [public_id(method)] if selector == "none" else [selector],
              "selective_method": public_id(method), "heuristic_method": public_id(method),
              "selector_baseline_first": selector != "none", "max_queries_per_dataset": None,
              "selector_model_path": str(model_path) if model_path else None,
              "cache_dir": str(folder/"live_cache")}
    return {"project_root": str(Path(__file__).resolve().parents[2]),
            "datasets": [d["slug"] for d in DATASETS if d["name"] in set(result["metrics"].dataset)],
            "query_split_file": str(split_path), "query_split": "test",
            "output_dir": str(folder/"live"), "checkpoint_dir": str(folder/"live_checkpoints"),
            "qexp": public_config(config)}


def run_matrix_study(run, output, *, methods=TECHNIQUES, selectors=("s1", "s2", "s3", "s4"),
                     estimators=ESTIMATORS, splits=None, neural_seeds=(42, 43, 44), feature_set="bge",
                     harm_penalty=1., cost_weight=.001, seed=42, neural_max_epochs=300,
                     neural_patience=30, resamples=2000, permutations=9999, plots=True):
    validate_options(methods, selectors, estimators, neural_seeds, feature_set)
    if (not np.isfinite([harm_penalty, cost_weight]).all() or min(harm_penalty, cost_weight) < 0
            or min(neural_max_epochs, neural_patience, resamples, permutations) < 1):
        raise ValueError("Parâmetros da matriz inválidos.")
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Use uma pasta nova; estudos anteriores são preservados.")
    result = load_result(run)
    config = {**QEXP_CONFIG, **result["metadata"]["config"], "selector_baseline_first": True}
    result["metadata"]["config"] = config
    methods = [internal_id(m) for m in methods]
    missing = set(methods) | {"B"}
    missing -= set(result["metrics"].pipeline_id)
    if missing:
        raise ValueError(f"Snapshot sem técnicas: {', '.join(sorted(map(public_id, missing)))}. Colete configs/matrix_full.json.")
    if set(methods) & FIXED_METHODS:
        verify_fixed_candidates(result, sorted(set(methods) & FIXED_METHODS))
    partition = pd.read_csv(splits, dtype=str, keep_default_na=False) if splits else make_splits(result, seed)
    output.mkdir(parents=True)
    split_path = output/"splits.csv"
    partition.to_csv(split_path, index=False)
    frame = utility_table(result, methods, split_path, selection_scope="all_techniques")
    required = set(FEATURE_SETS[feature_set]) if "s3" in selectors else set()
    if "s1" in selectors or "s2" in selectors:
        required.update(["relative_gap", "score_entropy", "lexical_overlap"])
    if "s2" in selectors:
        required.add("rank_divergence")
    if not required <= set(frame):
        raise ValueError(f"Snapshot sem atributos: {sorted(required-set(frame))}; não serão inventados.")
    metrics = result["metrics"].set_index(["dataset", "query_id", "pipeline_id"])
    artifacts = {(a["dataset"], a["query_id"]): a for a in result["artifacts"] if a["pipeline_id"] == "B"}
    # Custos de rotas seletivas incluem o BGE original; técnica uniforme mantém seu custo próprio.
    for m in methods:
        frame[f"uniform_sec_{m}"] = frame[f"seconds_{m}"]
        if m == "A":
            frame[f"seconds_{m}"] = frame.seconds_B
        elif m in {"C", "C_H4", "D", "F"}:
            extras = []
            for r in frame.itertuples():
                base, target = metrics.loc[(r.dataset, r.query_id, "B")], metrics.loc[(r.dataset, r.query_id, m)]
                # Falhas C/D já reutilizam B. t4 ainda faz uma passagem adicional.
                reused_base = m in {"C", "D"} and not target.expansion_applied
                extra = 0. if reused_base else float(base.rerank_sec)
                if m in {"C", "C_H4"} and target.expansion_applied:
                    extra += float(base.retrieval_sec)
                extras.append(extra)
            frame[f"seconds_{m}"] += extras
        frame[f"extra_sec_{m}"] = np.maximum(frame[f"seconds_{m}"]-frame.seconds_B, 0)
    images = run.parent.parent/"images" if run.parent.name == "outputs" and output.parent == run.parent.parent else output/"images"
    prefix = output.name+"_"
    manifest = {"status": "training", "source": str(run), "source_sha256": hashlib.sha256(run.read_bytes()).hexdigest(),
                "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))},
                "techniques": list(map(public_id, methods)), "selectors": list(selectors), "estimators": list(estimators),
                "neural_seeds": list(neural_seeds), "feature_set": feature_set, "seed": seed,
                "harm_penalty": harm_penalty, "cost_weight": cost_weight, "images": str(images),
                "partition_counts": frame.groupby(["dataset", "split"]).size().rename("queries").reset_index().to_dict("records"),
                "scope": "Exploratório. t1/t2 são controles, t3–t6 podem mudar candidatos, t7/t8/Q mantêm candidatos de t2.",
                "cost": "Replay composto: rotas seletivas incluem BGE original e atributos; exclui nova latência de decisão. Não somar linhas como custo de execução.",
                "statistics": "Contrastes pareados por grupo, cada célula vs t2 e técnica uniforme; Holm conjunto em matrix_contrasts.csv. Sementes não são observações independentes."}
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    policies = []
    for method in methods:
        specs = [("none", "none", None), *[(s, s, None) for s in selectors if s != "s3"]]
        if "s3" in selectors:
            specs += [(f"s3_{e}", "s3", None) for e in estimators if e != "neural"]
            if "neural" in estimators:
                specs += [(f"s3_neural_seed_{s}", "s3", s) for s in neural_seeds]
        for variant, selector, neural_seed in specs:
            name = f"{public_id(method)}__{variant}"
            print(f"Matriz: {name} — treino/validação", flush=True)
            model, artifact, thresholds = None, None, None
            if selector == "s3":
                if variant == "s3_logistic":
                    artifact, thresholds, _ = fit_selector(result, split_path, method, feature_set, allow_constant=True)
                    model = LearnedSelector(artifact)
                elif variant in {"s3_gain", "s3_weighted_logistic"}:
                    artifact, thresholds = fit_benefit_selector(result, split_path, [method], feature_set,
                                                                estimator="ridge" if variant == "s3_gain" else "weighted_logistic", frame=frame)
                    model = BenefitSelector(artifact)
                elif variant == "s3_utility":
                    artifact, thresholds = fit_utility_selector(result, split_path, [method], feature_set,
                                                                harm_penalty, cost_weight, frame=frame, selection_scope="all_techniques")
                    model = UtilitySelector(artifact)
                else:
                    artifact, thresholds = fit_neural_selector(result, split_path, [method], feature_set,
                                                               harm_penalty, cost_weight, frame=frame, seed=neural_seed,
                                                               max_epochs=neural_max_epochs, patience=neural_patience,
                                                               selection_scope="all_techniques")
                    model = NeuralUtilitySelector(artifact)
            folder = output/"policies"/name
            folder.mkdir(parents=True)
            model_path = folder/"selector.json" if model else None
            if model:
                artifact.update(policy_name=name, source_sha256=manifest["source_sha256"])
                model_path.write_text(json.dumps(public_model(artifact), indent=2, ensure_ascii=False, allow_nan=False))
                thresholds.to_csv(folder/"validation_thresholds.csv", index=False)
                if "training_history" in artifact:
                    pd.DataFrame(artifact["training_history"]).to_csv(folder/"training_history.csv", index=False)
            profile = _profile(result, split_path, folder, method, selector, model_path)
            (folder/"run_config.json").write_text(json.dumps(profile, indent=2, ensure_ascii=False))
            policies.append((method, variant, selector, model))
    # Nenhum resultado de teste é usado até todos os modelos e limiares estarem congelados.
    rows = []
    for method, variant, selector, model in policies:
        for r in frame[frame.split == "test"].to_dict("records"):
            extra = artifacts[(r["dataset"], r["query_id"])]
            overhead, score = 0., 0.
            if selector == "none":
                action = method
            elif selector in {"s1", "s2"}:
                chosen, _ = qexp_should_expand(r, config, r["rank_divergence"] if selector == "s2" else None)
                action = method if chosen else "B"
                overhead = extra.get("feature_lexical_sec", 0.) + (extra.get("feature_bge_sec", 0.) if selector == "s2" else 0.)
            elif selector == "s4":
                value = hashlib.sha256(json.dumps([config["seed"], r["dataset"], str(r["query_id"])]).encode()).hexdigest()
                action = method if int(value[:16], 16)/2**64 < config["random_expansion_rate"] else "B"
            else:
                model.check_unseen(r["dataset"], r["query_id"], r["query_text"])
                action, score = model.choose(r)
                overhead = extra.get("feature_lexical_sec", 0.)
                if uses_bge(feature_set):
                    overhead += extra.get("feature_bge_sec", 0.)
                if uses_rarity(feature_set):
                    overhead += extra.get("feature_corpus_sec", 0.)
            quality = r["baseline_ndcg"] if action == "B" else r[f"ndcg_{action}"]
            seconds = r[f"uniform_sec_{method}"] if selector == "none" else r[f"seconds_{action}"]+overhead
            scope = "reference_control" if method in {"A", "B"} else "fixed_candidates" if method in FIXED_METHODS else "retrieval_changes"
            rows.append({"dataset": r["dataset"], "query_id": r["query_id"], "group": r["group"],
                         "technique": public_id(method), "selector": selector, "variant": variant, "scope": scope,
                         "selected_method": public_id(action), "decision_score": score,
                         "selected_ndcg": quality, "baseline_ndcg": r["baseline_ndcg"], "uniform_ndcg": r[f"ndcg_{method}"],
                         "delta_vs_t2": quality-r["baseline_ndcg"], "delta_vs_uniform": quality-r[f"ndcg_{method}"],
                         "selected_technique": action == method, "deviates_from_t2": action != "B",
                         "expansion_selected": action not in {"A", "B"},
                         "estimated_policy_sec": seconds, "baseline_sec": r["seconds_B"],
                         "uniform_sec": r[f"uniform_sec_{method}"]})
    predictions = pd.DataFrame(rows)
    predictions.to_csv(output/"matrix_predictions.csv", index=False)
    summaries, contrasts = [], []
    for (dataset, technique, variant), g in predictions.groupby(["dataset", "technique", "variant"], sort=False):
        stats = paired_statistics(g.delta_vs_t2, g.group, seed=stable_seed(seed, f"{dataset}/{technique}/{variant}"),
                                  resamples=resamples, permutations=permutations)
        summaries.append({"dataset": dataset, "technique": technique, "variant": variant, "selector": g.selector.iloc[0],
                          "scope": g.scope.iloc[0], **stats, "ndcg": g.selected_ndcg.mean(),
                          "delta_vs_uniform": g.delta_vs_uniform.mean(), "expansion_rate": g.expansion_selected.mean(),
                          "selection_rate": g.selected_technique.mean(), "estimated_sec": g.estimated_policy_sec.mean(),
                          "harm_rate": (g.delta_vs_t2 < -1e-12).mean(),
                          "utility": (g.delta_vs_t2-harm_penalty*np.maximum(-g.delta_vs_t2, 0)
                                      -cost_weight*np.maximum(g.estimated_policy_sec-g.baseline_sec, 0)).mean()})
        for reference, delta in [("t2", g.delta_vs_t2)] + ([] if variant == "none" else [("uniform", g.delta_vs_uniform)]):
            comparison = paired_statistics(delta, g.group, seed=stable_seed(seed, f"{dataset}/{technique}/{variant}/{reference}"),
                                           resamples=resamples, permutations=permutations)
            contrasts.append({"dataset": dataset, "technique": technique, "variant": variant, "reference": reference, **comparison})
    summary, contrasts = pd.DataFrame(summaries), pd.DataFrame(contrasts)
    contrasts["p_holm"] = holm(contrasts.p_value)
    summary.to_csv(output/"matrix_summary.csv", index=False)
    contrasts.to_csv(output/"matrix_contrasts.csv", index=False)
    pd.DataFrame([{"technique": public_id(m), "description": QEXP_PIPELINES[m], "variant": v, "selector": s,
                   "run_config": f"policies/{public_id(m)}__{v}/run_config.json"} for m,v,s,_ in policies]).to_csv(output/"matrix_design.csv", index=False)
    if plots:
        render_matrix(summary, images, prefix)
    lines = ["# Matriz de técnicas e seletores", "", manifest["scope"], "", manifest["cost"], "", manifest["statistics"], "",
             "t1 é controle de escolha BM25/BGE; t2 é identidade. Nenhum deles representa expansão. "
             "s1/s2 usam regras fixas; s3 distingue estimadores; s4 é controle aleatório. Sem seleção: none.", "",
             "Todas as políticas usam as mesmas partições. Modelos e limiares congelados antes de avaliar teste. "
             "Sementes são reportadas separadamente, sem escolher pelo teste. A comparação é por técnica; não é um novo portfólio aprendido.", "",
             summary.to_markdown(index=False), "", "Os perfis em policies/*/run_config.json executam uma célula no teste. "
             "Use sessão nova para medir policy_wall_sec. A análise local não consome GPU."]
    if plots:
        import os
        lines += ["", f"[Gráficos]({os.path.relpath(images/f'{prefix}gallery.html', output)})"]
    (output/"report.md").write_text('\n'.join(lines)+'\n', encoding="utf-8")
    manifest.update(status="completed", cells=len(policies))
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return output


def render_matrix(summary, images, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    images.mkdir(parents=True, exist_ok=True)
    if any(images.glob(prefix+"matrix_*.pdf")):
        raise ValueError("Figuras da matriz já existem; escolha outra saída.")
    names = []
    for metric, title, diverging in [("mean_delta", "Delta nDCG vs t2", True),
                                     ("delta_vs_uniform", "Delta nDCG vs técnica sem seletor", True),
                                     ("estimated_sec", "Custo histórico composto (s)", False),
                                     ("expansion_rate", "Frequência de expansão", False)]:
        datasets = sorted(summary.dataset.unique())
        fig, axes = plt.subplots(1, len(datasets), figsize=(max(8, summary.variant.nunique())*len(datasets), 7), squeeze=False)
        bound = max(float(summary[metric].abs().max()), 1e-6)
        for ax, dataset in zip(axes[0], datasets):
            grid = summary[summary.dataset == dataset].pivot(index="technique", columns="variant", values=metric)
            grid = grid.reindex([m for m in TECHNIQUES if m in grid.index])
            values = grid.to_numpy()
            im = ax.imshow(values, cmap="RdBu" if diverging else "viridis", vmin=-bound if diverging else 0, vmax=bound, aspect="auto")
            ax.set_xticks(np.arange(len(grid.columns)), grid.columns, rotation=60, ha="right")
            ax.set_yticks(np.arange(len(grid.index)), grid.index)
            ax.set_title(dataset+" — "+title)
            for i in range(len(grid)):
                for j in range(len(grid.columns)):
                    ax.text(j, i, f"{values[i,j]:.3f}", ha="center", va="center", fontsize=8,
                            color="white" if (abs(values[i,j]) > bound*.6 if diverging else values[i,j] < bound*.5) else "black")
            fig.colorbar(im, ax=ax, shrink=.75)
        fig.tight_layout()
        name = prefix+"matrix_"+metric
        for ext in ("png", "pdf", "svg"):
            fig.savefig(images/f"{name}.{ext}", bbox_inches="tight", dpi=150)
        plt.close(fig)
        names.append(name)
    (images/f"{prefix}gallery.html").write_text('<!doctype html><meta charset="utf-8"><title>Matriz</title>'+''.join(
        f'<h2>{n}</h2><a href="{n}.pdf">PDF</a><br><img style="max-width:100%" src="{n}.svg">' for n in names), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--splits', type=Path)
    p.add_argument('--methods', nargs='+', default=TECHNIQUES)
    p.add_argument('--selectors', nargs='+', default=['s1', 's2', 's3', 's4'])
    p.add_argument('--estimators', nargs='+', default=ESTIMATORS)
    p.add_argument('--feature-set', choices=sorted(FEATURE_SETS), default='bge')
    p.add_argument('--neural-seeds', nargs='+', type=int, default=[42, 43, 44])
    p.add_argument('--neural-max-epochs', type=int, default=300)
    p.add_argument('--neural-patience', type=int, default=30)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--harm-penalty', type=float, default=1.)
    p.add_argument('--cost-weight', type=float, default=.001)
    p.add_argument('--resamples', type=int, default=2000)
    p.add_argument('--permutations', type=int, default=9999)
    p.add_argument('--no-plots', action='store_true')
    args = vars(p.parse_args())
    args['plots'] = not args.pop('no_plots')
    print(run_matrix_study(**args))


if __name__ == '__main__':
    main()
