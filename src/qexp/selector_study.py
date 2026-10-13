"""Estudo reproduzível de seletores sobre rankings salvos, sem executar GPU.

python -m qexp.selector_study --run outputs/qexp_run.json --output results/novo_estudo
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .benefit_selection import BenefitSelector, fit_benefit_selector, macro_mean, policy_table
from .config import QEXP_CONFIG, DATASETS, FIXED_METHODS
from .learned_selection import LearnedSelector, fit_selector, make_splits
from .selector_features import query_hash, uses_bge, uses_rarity
from .statistics import paired_statistics, holm, stable_seed

SCOPES = {"fixed": ("G", "K"), "retrieval": ("C", "D", "F"), "all": ("C", "D", "F", "G", "K")}


def make_domain_splits(result, dataset, seed=42):
    """Coleção inteira em teste; duplicatas textuais acompanham o teste."""
    base = result["metrics"].query("pipeline_id == 'B'")
    if dataset not in set(base.dataset):
        raise ValueError("Coleção de teste inexistente no snapshot.")
    frame = base[["dataset", "query_id", "query_text"]].copy()
    frame["group"] = frame.query_text.map(query_hash)
    heldout = set(frame.loc[frame.dataset == dataset, "group"])
    groups = np.array(sorted(set(frame.group) - heldout))
    if len(groups) < 2:
        raise ValueError("Faltam grupos independentes para treino e validação.")
    np.random.default_rng(seed).shuffle(groups)
    n_train = min(len(groups) - 1, max(1, int(.75 * len(groups))))
    training = set(groups[:n_train])
    frame["split"] = frame.group.map(lambda g: "test" if g in heldout else "train" if g in training else "validation")
    return frame[["dataset", "query_id", "split", "group"]]


def replay_policy(result, model, frame, name):
    """Qualidade e custo histórico composto; não mede latência de produção."""
    a = model.artifact
    model.validate_protocol({**QEXP_CONFIG, **result["metadata"]["config"], "selective_method": a["method"]})
    methods = a.get("methods", [a["method"]])
    test = frame[frame.split == "test"].copy()
    expected = {tuple(pair) for pair in a["test_ids"]}
    if set(map(tuple, test[["dataset", "query_id"]].values)) != expected:
        raise ValueError("Consultas de teste diferentes do manifesto do modelo.")
    metrics = result["metrics"].set_index(["dataset", "query_id", "pipeline_id"])
    artifacts = {(r["dataset"], r["query_id"]): r for r in result["artifacts"] if r["pipeline_id"] == "B"}
    validation = frame[frame.split == "validation"]
    # Melhor política uniforme escolhida antes de consultar o teste.
    utilities = {"B": 0.0, **{m: macro_mean(validation[f"delta_{m}"], validation.dataset) for m in methods}}
    uniform = max(utilities, key=utilities.get)
    rows = []
    for row in test.to_dict("records"):
        key = (row["dataset"], row["query_id"])
        model.check_unseen(*key, row["query_text"])
        action, score = model.choose(row)
        base, target = metrics.loc[(*key, "B")], metrics.loc[(*key, action)]
        info = artifacts[key]
        cost = float(target.estimated_uncached_sec) + info.get("feature_lexical_sec", 0)
        if uses_bge(a["feature_set"]):
            cost += info.get("feature_bge_sec", 0)
            if action == "F" or (action in {"C", "D"} and target.expansion_applied):
                cost += float(base.rerank_sec)
        if uses_rarity(a["feature_set"]):
            cost += info.get("feature_corpus_sec", 0)
        if action == "C" and target.expansion_applied:
            cost += float(base.retrieval_sec)
        selected = row["baseline_ndcg"] if action == "B" else row[f"ndcg_{action}"]
        oracle = max([row["baseline_ndcg"], *[row[f"ndcg_{m}"] for m in methods]])
        scores = model.scores(row) if isinstance(model, BenefitSelector) else {methods[0]: score}
        rows.append({**row, "policy": name, "selected_method": action, "decision_score": score,
                     **{f"score_{m}": value for m, value in scores.items()},
                     "expand": action != "B", "selected_ndcg": selected,
                     "policy_delta": selected - row["baseline_ndcg"], "oracle_ndcg": oracle,
                     "oracle_regret": oracle - selected, "uniform_reference": uniform,
                     "uniform_ndcg": row["baseline_ndcg"] if uniform == "B" else row[f"ndcg_{uniform}"],
                     "estimated_policy_sec": cost, "baseline_sec": float(base.estimated_uncached_sec),
                     "expansion_applied": bool(target.expansion_applied),
                     "generation_status_selected": target.generation_status,
                     "scope": "fixed" if set(methods) <= FIXED_METHODS else "retrieval" if set(methods) <= {"C", "D", "F"} else "mixed"})
    return pd.DataFrame(rows)


def summarize_policies(predictions, *, seed=42, resamples=2000, permutations=9999):
    summaries, contrasts, random_records = [], [], []
    for (name, dataset), group in predictions.groupby(["policy", "dataset"], sort=True):
        key_seed = stable_seed(seed, f"{name}/{dataset}")
        stats = paired_statistics(group.policy_delta, group.group, seed=key_seed,
                                  resamples=resamples, permutations=permutations)
        contrasts.append({"policy": name, "dataset": dataset, "reference": "B", **stats})
        uniform = group.uniform_reference.iloc[0]
        if uniform != "B":
            contrasts.append({"policy": name, "dataset": dataset, "reference": uniform,
                              **paired_statistics(group.selected_ndcg - group.uniform_ndcg, group.group,
                                                  seed=key_seed, resamples=resamples, permutations=permutations)})
        # Mesmas quantidades de B/C/D/F/G/K por coleção; embaralha as ações.
        rng = np.random.default_rng(key_seed)
        actions = group.selected_method.to_numpy()
        quality = {"B": group.baseline_ndcg.to_numpy(),
                   **{c[5:]: group[c].to_numpy() for c in group if c.startswith("ndcg_")}}
        random_means = []
        for trial in range(500):
            shuffled = rng.permutation(actions)
            values = np.array([quality[m][i] for i, m in enumerate(shuffled)])
            random_means.append(values.mean())
            random_records.append({"policy": name, "dataset": dataset, "trial": trial, "ndcg": values.mean()})
        baseline, selected = group.baseline_ndcg.mean(), group.selected_ndcg.mean()
        summaries.append({"policy": name, "dataset": dataset, "scope": group.scope.iloc[0], **stats,
                          "baseline_ndcg": baseline, "selected_ndcg": selected,
                          "uniform_reference": uniform, "uniform_ndcg": group.uniform_ndcg.mean(),
                          "oracle_ndcg": group.oracle_ndcg.mean(), "oracle_regret": group.oracle_regret.mean(),
                          "expansion_rate": group.expand.mean(), "applied_rate": group.expansion_applied.mean(),
                          "missed_gain": np.maximum(group.oracle_ndcg - group.selected_ndcg, 0).mean(),
                          "estimated_policy_sec": group.estimated_policy_sec.mean(),
                          "baseline_sec": group.baseline_sec.mean(),
                          "random_ndcg_mean": float(np.mean(random_means)),
                          "random_ndcg_p025": float(np.quantile(random_means, .025)),
                          "random_ndcg_p975": float(np.quantile(random_means, .975)),
                          "random_percentile": float(np.mean(np.asarray(random_means) <= selected))})
    contrasts = pd.DataFrame(contrasts)
    contrasts["p_holm"] = holm(contrasts.p_value)
    summary = pd.DataFrame(summaries).merge(contrasts.query("reference == 'B'")[["policy", "dataset", "p_holm"]],
                                           on=["policy", "dataset"], validate="one_to_one")
    return summary, contrasts, pd.DataFrame(random_records)


def pipeline_diagnostics(result, frame, *, seed=42, resamples=2000, permutations=9999):
    """Mesma partição de teste para gráficos das alternativas uniformes."""
    test = frame.loc[frame.split == "test", ["dataset", "query_id", "group"]]
    metrics = result["metrics"].merge(test, on=["dataset", "query_id"], validate="many_to_one")
    k = result["metadata"]["config"]["eval_k"]
    base = metrics.query("pipeline_id == 'B'")[["dataset", "query_id", f"ndcg@{k}"]].rename(columns={f"ndcg@{k}": "baseline_ndcg"})
    metrics = metrics.merge(base, on=["dataset", "query_id"], validate="many_to_one")
    metrics["delta"] = metrics[f"ndcg@{k}"] - metrics.baseline_ndcg
    summaries = []
    for (dataset, method), group in metrics.groupby(["dataset", "pipeline_id"]):
        stats = paired_statistics(group.delta, group.group, seed=stable_seed(seed, f"{dataset}/{method}"),
                                  resamples=resamples, permutations=permutations)
        summaries.append({"dataset": dataset, "pipeline_id": method, **stats,
                          "ndcg": group[f"ndcg@{k}"].mean(), "baseline_ndcg": group.baseline_ndcg.mean(),
                          "estimated_sec": group.estimated_uncached_sec.mean()})
    summary = pd.DataFrame(summaries)
    nonbase = summary.pipeline_id != "B"
    summary["p_holm"] = 1.0
    summary.loc[nonbase, "p_holm"] = holm(summary.loc[nonbase, "p_value"])
    return summary, metrics


def run_study(run, output, *, splits=None, test_dataset=None, seed=42, resamples=2000,
              permutations=9999, harm_penalty=0.0, max_expansion_rate=1.0, plots=True):
    from .checkpoints import load_result
    output, run = Path(output).resolve(), Path(run).resolve()
    if output.exists():
        raise ValueError("Use uma pasta nova; resultados anteriores não são sobrescritos.")
    if splits and test_dataset:
        raise ValueError("Use --splits ou --test-dataset, não ambos.")
    result = load_result(run)
    split_frame = (pd.read_csv(splits, dtype=str, keep_default_na=False) if splits else
                   make_domain_splits(result, test_dataset, seed) if test_dataset else make_splits(result, seed))
    # Valida todos os métodos e a ausência de vazamento antes de treinar.
    output.mkdir(parents=True)
    split_path = output / "splits.csv"
    split_frame.to_csv(split_path, index=False)
    frame = policy_table(result, SCOPES["all"], split_path)
    if "group" not in frame:
        frame["group"] = frame.query_hash
    source_hash = hashlib.sha256(run.read_bytes()).hexdigest()
    variants = [(f"{kind}_{m}_{features}", (m,), features, kind)
                for m in SCOPES["all"] for features in ("lexical", "bge")
                for kind in ("logistic", "weighted_logistic", "ridge")]
    variants += [(f"ridge_{scope}_{features}", methods, features, "ridge")
                 for scope, methods in SCOPES.items() for features in ("lexical", "bge")]
    manifest = {"schema": 1, "source": str(run), "source_sha256": source_hash,
                "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(Path(__file__).parent.glob("*.py"))},
                "seed": seed, "resamples": resamples, "permutations": permutations,
                "harm_penalty": harm_penalty, "max_expansion_rate": max_expansion_rate,
                "test_dataset": test_dataset, "status": "running",
                "variants": [v[0] for v in variants], "metric": f"ndcg@{result['metadata']['config']['eval_k']}",
                "scope": "Exploratório: consultas já analisadas; não constitui teste externo inédito.",
                "statistics": "Bootstrap e permutação pareados por grupo dentro de coleção; Holm global por tabela de contrastes.",
                "cost": "Replay de custos históricos compostos; inclui atributos e BGE inicial necessário, exclui nova latência da decisão."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    models, choices, threshold_rows = {}, [], []
    for name, methods, features, kind in variants:
        print(f"Treinando {name} (somente treino/validação)", flush=True)
        if kind == "logistic":
            artifact, thresholds, _ = fit_selector(result, split_path, methods[0], features)
            # A referência antiga mantém seu próprio critério de ajuste.
            validation = frame[frame.split == "validation"]
            model = LearnedSelector(artifact)
            gains = [r[f"delta_{methods[0]}"] * model.predict(r)[0] for r in validation.to_dict("records")]
            validation_gain = macro_mean(gains, validation.dataset)
        else:
            artifact, thresholds = fit_benefit_selector(result, split_path, methods, features, kind,
                                                        harm_penalty, max_expansion_rate, seed=seed, frame=frame)
            model = BenefitSelector(artifact)
            validation_gain = artifact["validation_choice"]["validation_gain"]
        artifact["source_sha256"] = source_hash
        artifact["policy_name"] = name
        folder = output / "models" / name
        folder.mkdir(parents=True)
        (folder / "selector.json").write_text(json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False))
        thresholds.to_csv(folder / "validation_thresholds.csv", index=False)
        threshold_rows.append(thresholds.assign(policy=name))
        scope = "fixed" if set(methods) <= FIXED_METHODS else "retrieval" if set(methods) <= {"C", "D", "F"} else "mixed"
        validation_rows = frame[frame.split == "validation"].to_dict("records")
        val_actions = [model.choose(r)[0] for r in validation_rows]
        val_gains = np.array([0.0 if action == "B" else r[f"delta_{action}"]
                             for action, r in zip(val_actions, validation_rows)])
        val_datasets = [r["dataset"] for r in validation_rows]
        max_rate = pd.Series([action != "B" for action in val_actions]).groupby(val_datasets).mean().max()
        choices.append({"policy": name, "scope": scope, "features": features, "estimator": kind,
                        "validation_gain": validation_gain,
                        "validation_utility": macro_mean(val_gains - harm_penalty * np.maximum(-val_gains, 0), val_datasets),
                        "eligible": bool(max_rate <= max_expansion_rate + 1e-12),
                        "methods": ",".join(methods)})
        models[name] = model
        # Perfil pronto para executar exclusivamente a política nas queries reservadas.
        project = Path(__file__).resolve().parents[2]
        profile = {"project_root": str(project), "datasets": [d["slug"] for d in DATASETS],
                   "query_split_file": str(split_path), "query_split": "test",
                   "output_dir": str(output / "live" / name),
                   "qexp": {**result["metadata"]["config"], "pipelines": ["L"],
                            "selective_method": artifact["method"], "selector_model_path": str(folder / "selector.json")}}
        profile["qexp"]["cache_dir"] = str(output / "live_cache")
        profile["qexp"]["max_queries_per_dataset"] = None
        (folder / "run_config.json").write_text(json.dumps(profile, indent=2, ensure_ascii=False))
    validation = pd.DataFrame(choices)
    # Destaques congelados na validação; não selecionados pelas tabelas de teste.
    champions = validation[validation.eligible].sort_values(["validation_utility", "policy"], ascending=[False, True]).groupby("scope", sort=True).head(1)
    validation.to_csv(output / "validation_models.csv", index=False)
    champions.to_csv(output / "validation_champions.csv", index=False)
    pd.concat(threshold_rows, ignore_index=True).to_csv(output / "validation_thresholds.csv", index=False)
    print("Modelos congelados. Avaliando uma vez as consultas reservadas...", flush=True)
    predictions = pd.concat([replay_policy(result, m, frame, name) for name, m in models.items()], ignore_index=True)
    summary, contrasts, randoms = summarize_policies(predictions, seed=seed, resamples=resamples, permutations=permutations)
    pipelines, deltas = pipeline_diagnostics(result, frame, seed=seed, resamples=resamples, permutations=permutations)
    for name, table in {"test_predictions": predictions, "test_summary": summary, "test_contrasts": contrasts,
                        "random_matched": randoms, "pipeline_summary": pipelines, "pipeline_deltas": deltas}.items():
        table.to_csv(output / f"{name}.csv", index=False)
    manifest.update(status="completed", champions=champions.policy.tolist(),
                    partition_counts=frame.groupby(["dataset", "split"]).size().rename("queries").reset_index().to_dict("records"))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    lines = ["# Seleção de expansão para melhorar o BGE", "", manifest["scope"], "",
             "## Protocolo", "", "BGE permanece nos dois caminhos. G/K preservam candidatos; C/D/F modificam recuperação.",
             "Treino ajusta coeficientes e escala; validação escolhe regularização e limiar. Teste não ajusta modelos.",
             "Ridge prevê ganho (ou ganho penalizado por perdas); logística ponderada prioriza exemplos de maior impacto.",
             "Os modelos novos equilibram as coleções; a logística antiga é a referência sem alteração.",
             "O limite de expansão é verificado em cada coleção da validação, não garante orçamento no teste/produção.",
             "", "## Partições", "", pd.DataFrame(manifest["partition_counts"]).to_markdown(index=False),
             "", "## Destaques escolhidos na validação", "", champions.to_markdown(index=False),
             "", "## Resultados reservados dos destaques", "",
             summary[summary.policy.isin(champions.policy)][["policy", "dataset", "baseline_ndcg", "selected_ndcg", "mean_delta", "ci95_low", "ci95_high", "p_holm", "expansion_rate", "harm_rate", "oracle_regret"]].to_markdown(index=False),
             "", "## Leitura estatística", "", manifest["statistics"],
             "Intervalos são marginais de 95%; p bilateral por troca de sinais (exato até 15 grupos com soma não nula).",
             "O teste assume permutabilidade dos métodos no grupo sob a hipótese nula. Holm inclui todas as políticas/coleções/referências da tabela.",
             "O oráculo consulta os resultados reais e é apenas um teto retrospectivo. Seu ganho não demonstra previsibilidade.",
             "O controle aleatório preserva exatamente as contagens de cada ação por coleção, em 500 sorteios; sua faixa é uma distribuição de políticas aleatórias, não IC da média.",
             "A família pipeline_summary é separada da família test_contrasts. Ausência de significância não demonstra equivalência.",
             "", "## Custo e limitações", "", manifest["cost"],
             "Falhas de geração permanecem na avaliação. applied_rate mede aplicações efetivas; expansion_rate mede solicitações.",
             "Os gráficos de limiar usam apenas validação. Comparar muitas versões neste estudo é exploração; confirmar em dados inéditos.",
             "", "## Arquivos", "", "CSV por consulta, contrastes, modelos JSON e configurações executáveis em models/*/run_config.json.",
             "Figuras em figures/ (PNG, SVG e PDF). Abra figures/gallery.html para navegar localmente."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if plots:
        from .visualization import render_study
        render_study(output)
    print(f"Estudo salvo em {output}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--test-dataset", choices=[d["name"] for d in DATASETS])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=9999)
    parser.add_argument("--harm-penalty", type=float, default=0.0)
    parser.add_argument("--max-expansion-rate", type=float, default=1.0)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    run_study(args.run, args.output, splits=args.splits, test_dataset=args.test_dataset, seed=args.seed,
              resamples=args.resamples, permutations=args.permutations, harm_penalty=args.harm_penalty,
              max_expansion_rate=args.max_expansion_rate, plots=not args.no_plots)


if __name__ == "__main__":
    main()
