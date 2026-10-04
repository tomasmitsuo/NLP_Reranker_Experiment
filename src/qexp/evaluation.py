import json
from pathlib import Path
import numpy as np
import pandas as pd

def qexp_hypothesis_tables(metrics, config):
    from scipy.stats import spearmanr, wilcoxon

    k, n = config["eval_k"], config["candidate_n"]
    ndcg, mrr, recall = f"ndcg@{k}", f"mrr@{k}", f"candidate_recall@{n}"
    rng = np.random.default_rng(config["seed"])
    comparisons, query_deltas, associations = [], [], []

    def paired(hypothesis, dataset, method, reference, metric, differences):
        differences = np.asarray(differences, dtype=float)
        bootstrap = np.array([rng.choice(differences, size=len(differences), replace=True).mean()
                              for _ in range(1000)])
        nonzero = np.abs(differences) > 1e-12
        comparisons.append({
            "hypothesis": hypothesis, "dataset": dataset, "method": method, "reference": reference,
            "metric": metric, "num_pairs": len(differences), "mean_delta": differences.mean(),
            "ci95_low": float(np.quantile(bootstrap, 0.025)),
            "ci95_high": float(np.quantile(bootstrap, 0.975)),
            "positive": int((differences > 1e-12).sum()),
            "ties": int((~nonzero).sum()), "negative": int((differences < -1e-12).sum()),
            "p_value": float(wilcoxon(differences, alternative="two-sided").pvalue) if nonzero.any() else 1.0,
        })

    for dataset, group in metrics.groupby("dataset"):
        methods = {method: frame.set_index("query_id").sort_index()
                   for method, frame in group.groupby("pipeline_id")}
        base = methods["B"]
        for method, frame in methods.items():
            if not frame.index.equals(base.index):
                raise ValueError(f"Queries não pareadas: {dataset}/{method}")
            if method in {"A", "B", "C_H4"}:
                continue
            delta_recall = frame[recall] - base[recall]
            delta_ndcg = frame[ndcg] - base[ndcg]
            paired("H1", dataset, method, "A", recall, frame[recall] - methods["A"][recall])
            paired("H2", dataset, method, "B", ndcg, delta_ndcg)
            paired("H2", dataset, method, "B", mrr, frame[mrr] - base[mrr])
            # Associação por query é descritiva: não identifica efeito causal do candidate recall.
            valid = len(frame) >= 3 and delta_recall.nunique() > 1 and delta_ndcg.nunique() > 1
            rho, p_value = spearmanr(delta_recall, delta_ndcg) if valid else (np.nan, np.nan)
            associations.append({
                "dataset": dataset, "method": method, "num_queries": len(frame),
                "spearman_delta_candidate_recall_vs_ndcg": rho, "p_value_unadjusted": p_value,
                "status": "ok" if valid else "insufficient_variation",
            })
            for query_id in frame.index:
                query_deltas.append({
                    "dataset": dataset, "query_id": query_id, "pipeline_id": method,
                    "query_text": frame.loc[query_id, "query_text"],
                    "delta_candidate_recall": delta_recall.loc[query_id],
                    "delta_ndcg": delta_ndcg.loc[query_id],
                    "delta_mrr": frame.loc[query_id, mrr] - base.loc[query_id, mrr],
                    "degraded_vs_B": bool(delta_ndcg.loc[query_id] < -1e-12),
                    "expansion_requested": bool(frame.loc[query_id, "expansion_requested"]),
                    "generation_status": frame.loc[query_id, "generation_status"],
                })
        for method in ("E", "E_RA"):
            if method not in methods:
                continue
            selective, always = methods[method], methods["D"]
            selective_harm = (selective[ndcg] < base[ndcg] - 1e-12).astype(int)
            always_harm = (always[ndcg] < base[ndcg] - 1e-12).astype(int)
            paired("H3", dataset, method, "D", "degradation_rate_vs_B", selective_harm - always_harm)
            paired("H3", dataset, method, "D", ndcg, selective[ndcg] - always[ndcg])
        if "C_H4" in methods:
            for metric in (ndcg, mrr):
                paired("H4", dataset, "C", "C_H4", metric, methods["C"][metric] - methods["C_H4"][metric])

    comparisons = pd.DataFrame(comparisons)
    # Holm controla as múltiplas comparações desta tabela; intervalos bootstrap são marginais.
    order = np.argsort(comparisons["p_value"].to_numpy())
    adjusted = np.maximum.accumulate(comparisons["p_value"].to_numpy()[order] * np.arange(len(order), 0, -1))
    comparisons["p_value_holm"] = np.nan
    comparisons.loc[comparisons.index[order], "p_value_holm"] = np.minimum(adjusted, 1.0)
    deltas = pd.DataFrame(query_deltas)
    degradation = deltas.groupby(["dataset", "pipeline_id"], as_index=False).agg(
        num_queries=("query_id", "size"), degradation_rate=("degraded_vs_B", "mean"),
        mean_delta_ndcg=("delta_ndcg", "mean"), mean_delta_candidate_recall=("delta_candidate_recall", "mean"),
    )
    return {"hypotheses": comparisons, "h2_associations": pd.DataFrame(associations),
            "query_deltas": deltas, "degradation": degradation}


def qexp_export(result, config, output_dir="."):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.update(qexp_hypothesis_tables(result["metrics"], config))
    for name in ("metrics", "summary", "rankings", "decisions", "hypotheses",
                 "h2_associations", "query_deltas", "degradation"):
        result[name].to_csv(output / f"qexp_{name}.csv", index=False)
    # Exporta prompts e saídas brutas para auditoria de drift e inspeção dos piores casos.
    for name in ("metadata", "artifacts", "generations"):
        (output / f"qexp_{name}.json").write_text(
            json.dumps(result[name], ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
    lines = [
        "# Expansão seletiva de consultas para reranqueamento neural",
        "", "## Protocolo", "",
        f"Candidatos: {config['candidate_n']}; avaliação: @{config['eval_k']}; reformulações: {config['num_reformulations']}.",
        "Query original no BGE em B–F; C_H4 modifica apenas o texto da query, mantendo os candidatos de C.",
        "E usa regras lexicais fixas; E_RA acrescenta divergência dos prefixos BM25–BGE.",
        "Qrels são usados somente nas métricas. Limiares são heurísticos, não calibrados no teste.",
        "", "## Qualidade e custo", "", result["summary"].to_markdown(index=False),
        "", result["metadata"]["time_protocol"],
        "Os tempos históricos do cache podem vir de outro hardware: consulte qexp_generations.json.",
        "O total de execução está em qexp_metadata.json; não some custos compostos para estimá-lo.",
        "", "## Hipóteses H1–H4", "", result["hypotheses"].to_markdown(index=False),
        "", "Delta = método menos referência. Em H3/degradation_rate_vs_B, valores negativos favorecem a seleção.",
        "Em H4, valores positivos favorecem a query original no BGE. H4 mede ganho/perda pareados, não robustez universal.",
        "IC95%: bootstrap pareado de queries, 1000 reamostragens. Teste: Wilcoxon bilateral; p ajustado por Holm na tabela completa.",
        "Amostras pequenas e muitas queries empatadas limitam a análise. Não há confirmação automática das hipóteses.",
        "", "## H2: associação descritiva", "", result["h2_associations"].to_markdown(index=False),
        "", "Correlação entre ganhos de recall de candidatos e nDCG não demonstra causalidade; p desta tabela não é ajustado.",
        "", "## Degradação em relação a B", "", result["degradation"].to_markdown(index=False),
        "", "## Auditoria e limitações", "",
        "Consulte qexp_decisions.csv (sinais/motivos), qexp_query_deltas.csv (ganhos/perdas),",
        "qexp_rankings.csv (candidatos e ranking final), qexp_artifacts.json (queries/pesos RM3) e qexp_generations.json (prompts/gerações/falhas).",
        "Candidate Recall usa apenas relevância julgada; documentos não julgados são tratados como não relevantes pelas métricas existentes.",
        "Query2doc é uma adaptação zero-shot com LLM local, não uma reprodução do modelo/prompt few-shot do artigo.",
        result["metadata"]["rm3_protocol"],
        "Erros de formato do LLM conservam o ranking original e contam como falhas, não como expansões bem-sucedidas.",
        "As saídas do LLM são hipóteses para busca, não afirmações científicas verificadas.",
    ]
    (output / "qexp_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
