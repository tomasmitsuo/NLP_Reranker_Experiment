import hashlib
import json
import platform
import time
from importlib.metadata import version as package_version
import pandas as pd
from tqdm.auto import tqdm
from .config import QEXP_PIPELINES
from .metrics import ndcg_at_k, mrr_at_k, recall_at_k, average_precision_at_k
from .retrieval import retrieve_bm25, tokenize, qexp_multi_candidates, qexp_rm3_weights, qexp_weighted_bm25
from .selection import qexp_validate_config, qexp_difficulty_signals, qexp_rank_divergence, qexp_should_expand
from .timing import qexp_timed
from .progress import report, active

def run_qexp_experiment(experiments, config, expansion_cache, rerank_fn):
    """Reavalia A–F sobre as mesmas queries; métricas antigas continuam independentes."""
    qexp_validate_config(config)
    metric_rows, ranking_records, decision_rows, artifact_rows, dataset_records = [], [], [], [], []
    experiment_start = time.perf_counter()
    n, k = config["candidate_n"], config["eval_k"]

    for case in experiments.values():
        dataset = case["config"]["name"]
        queries = list(case["queries"].items())
        if config["max_queries_per_dataset"] is not None:
            queries = queries[:config["max_queries_per_dataset"]]
        dataset_records.append({
            "dataset": dataset, "source": case["config"], "documents": len(case["doc_ids"]),
            "query_ids": [str(qid) for qid, _ in queries],
            "queries_sha256": hashlib.sha256(json.dumps(queries, ensure_ascii=False).encode()).hexdigest(),
            "bm25_k1": case["bm25"].k1, "bm25_b": case["bm25"].b,
        })

        for query_id, query in tqdm(queries, desc=f"{dataset} | expansão A–F", disable=active()):
            query_id = str(query_id)
            report("A: recuperação BM25", dataset=dataset, query_id=query_id)
            initial, initial_sec = qexp_timed(retrieve_bm25, query, case["bm25"], case["doc_ids"], top_k=n)

            def record_result(pipeline_id, candidates, final, *, retrieval_sec, rerank_sec=0.0,
                              decision_sec=0.0, initial_rerank_sec=0.0, generation=None,
                              expansion_requested=False, expansion_applied=False, rerank_query=None):
                # Qrels entram apenas aqui, após decidir/recuperar/reranquear.
                relevant = case["qrels"].get(query_id, {})
                candidate_ids = [item["doc_id"] for item in candidates]
                final_ids = [item["doc_id"] for item in final]
                original_ids = {item["doc_id"] for item in initial}
                rel_ids = {doc_id for doc_id, rel in relevant.items() if rel > 0}
                generation_sec = generation["generation_sec"] if generation else 0.0
                generation_lookup_sec = generation["lookup_wall_sec"] if generation else 0.0
                other_sec = retrieval_sec + rerank_sec + decision_sec + initial_rerank_sec
                metric_rows.append({
                    "dataset": dataset, "query_id": query_id, "query_text": query,
                    "pipeline_id": pipeline_id, "pipeline": QEXP_PIPELINES[pipeline_id],
                    f"ndcg@{k}": ndcg_at_k(final_ids, relevant, k),
                    f"mrr@{k}": mrr_at_k(final_ids, relevant, k),
                    f"recall@{k}": recall_at_k(final_ids, relevant, k),
                    f"map@{k}": average_precision_at_k(final_ids, relevant, k),
                    f"candidate_recall@{n}": recall_at_k(candidate_ids, relevant, n),
                    "candidate_count": len(candidate_ids), "judged_relevant_count": len(rel_ids),
                    "relevant_candidates_gained": len((set(candidate_ids) - original_ids) & rel_ids),
                    "relevant_candidates_lost": len((original_ids - set(candidate_ids)) & rel_ids),
                    "expansion_requested": expansion_requested, "expansion_applied": expansion_applied,
                    "generation_status": generation["status"] if generation else "not_requested",
                    "generation_cache_hit": generation["cache_hit"] if generation else False,
                    "cache_key": generation["cache_key"] if generation else "",
                    "retrieval_sec": retrieval_sec, "rerank_sec": rerank_sec,
                    "decision_sec": decision_sec, "initial_rerank_sec": initial_rerank_sec,
                    "generation_sec": generation_sec, "generation_lookup_sec": generation_lookup_sec,
                    # Custos compostos: cada política paga todos os estágios que precisaria executar,
                    # mesmo se A/B já os calcularam. Cache não torna a geração gratuita na comparação.
                    "estimated_uncached_sec": other_sec + generation_sec,
                    "composed_observed_sec": other_sec + generation_lookup_sec,
                })
                for stage, ranking in (("candidates", candidates), ("final", final[:k])):
                    for rank, item in enumerate(ranking, start=1):
                        ranking_records.append({
                            "dataset": dataset, "query_id": query_id, "pipeline_id": pipeline_id,
                            "stage": stage, "rank": rank, "doc_id": item["doc_id"],
                            "score": item["score"], "relevance": relevant.get(item["doc_id"], 0),
                        })
                artifact_rows.append({
                    "dataset": dataset, "query_id": query_id, "pipeline_id": pipeline_id,
                    "original_query": query,
                    "reranker_query": rerank_query if rerank_query is not None else query,
                    "cache_key": generation["cache_key"] if generation else None,
                    "expansions": generation["expansions"] if generation else [],
                })

            record_result("A", initial, initial, retrieval_sec=initial_sec)
            report("B: reranqueamento BGE original")
            base_final, base_rerank_sec = qexp_timed(rerank_fn, query, initial, case["documents"])
            record_result("B", initial, base_final, retrieval_sec=initial_sec, rerank_sec=base_rerank_sec)

            # C: o pseudo-documento modifica somente a busca lexical.
            report("C: expansão Query2doc")
            generated = expansion_cache.get(query, "query2doc")
            expanded_query = (" ".join([query] * config["query_repetitions"] + generated["expansions"])
                              if generated["expansions"] else query)
            report("C: recuperação expandida")
            candidates_c, retrieve_c_sec = qexp_timed(
                retrieve_bm25, expanded_query, case["bm25"], case["doc_ids"], top_k=n
            )
            report("C: reranqueamento BGE")
            final_c, rerank_c_sec = qexp_timed(rerank_fn, query, candidates_c, case["documents"])
            record_result("C", candidates_c, final_c, retrieval_sec=retrieve_c_sec,
                          rerank_sec=rerank_c_sec, generation=generated, expansion_requested=True,
                          expansion_applied=bool(generated["expansions"]))
            if config["run_h4"]:
                # H4 isola o texto usado pelo BGE: nenhum candidato é adicionado/removido.
                report("C_H4: BGE com query expandida")
                final_h4, rerank_h4_sec = qexp_timed(rerank_fn, expanded_query, candidates_c, case["documents"])
                record_result("C_H4", candidates_c, final_h4, retrieval_sec=retrieve_c_sec,
                              rerank_sec=rerank_h4_sec, generation=generated, expansion_requested=True,
                              expansion_applied=bool(generated["expansions"]), rerank_query=expanded_query)

            signals, feature_sec = qexp_timed(qexp_difficulty_signals, query, initial, case["documents"], config)
            multi_result = None
            multi_generation = None
            policies = ["E"] + (["E_RA"] if config["run_reranker_aware"] else []) + ["D"]
            for pipeline_id in policies:
                report(f"{pipeline_id}: decisão e expansão Multi-query")
                divergence, divergence_sec = None, 0.0
                initial_bge_sec = 0.0
                if pipeline_id == "E_RA":
                    divergence, divergence_sec = qexp_timed(
                        qexp_rank_divergence, initial, base_final, config["signal_top_k"]
                    )
                    initial_bge_sec = base_rerank_sec
                if pipeline_id == "D":
                    expand, reason, decision_sec = True, "always", 0.0
                else:
                    (expand, reason), rule_sec = qexp_timed(qexp_should_expand, signals, config, divergence)
                    decision_sec = feature_sec + divergence_sec + rule_sec
                    decision_rows.append({
                        "dataset": dataset, "query_id": query_id, "pipeline_id": pipeline_id,
                        **signals, "rank_divergence": divergence, "expand": expand, "reason": reason,
                    })
                # A decisão antecede o acesso ao gerador. Uma query fácil em E não chama o LLM.
                generation = None
                if expand:
                    if multi_generation is None:
                        multi_generation = expansion_cache.get(query, "multi")
                    generation = multi_generation
                applied = bool(generation and generation["expansions"])
                candidates, retrieval_sec = initial, initial_sec
                final, final_sec = base_final, base_rerank_sec
                if applied:
                    if multi_result is None:
                        report(f"{pipeline_id}: recuperação Multi-query/RRF")
                        candidates, multi_sec = qexp_timed(
                            qexp_multi_candidates, query, generation["expansions"], initial, case, config
                        )
                        report(f"{pipeline_id}: reranqueamento BGE Multi-query")
                        final, final_sec = qexp_timed(rerank_fn, query, candidates, case["documents"])
                        multi_result = candidates, multi_sec, final, final_sec
                    candidates, multi_sec, final, final_sec = multi_result
                    retrieval_sec += multi_sec
                elif pipeline_id == "E_RA":
                    # BGE inicial já é o ranking final: não contabilizar a mesma chamada duas vezes.
                    final_sec = 0.0
                record_result(pipeline_id, candidates, final, retrieval_sec=retrieval_sec,
                              rerank_sec=final_sec, initial_rerank_sec=initial_bge_sec,
                              decision_sec=decision_sec, generation=generation,
                              expansion_requested=expand, expansion_applied=applied)

            # F: feedback lexical clássico, sem geração neural nem qrels.
            report("F: feedback RM3 e recuperação")
            weights, feedback_sec = qexp_timed(qexp_rm3_weights, query, initial, case["documents"], config)
            candidates_f, retrieval_f_sec = qexp_timed(
                qexp_weighted_bm25, weights, case["bm25"], case["doc_ids"], n
            )
            report("F: reranqueamento BGE")
            final_f, rerank_f_sec = qexp_timed(rerank_fn, query, candidates_f, case["documents"])
            record_result("F", candidates_f, final_f, retrieval_sec=initial_sec + feedback_sec + retrieval_f_sec,
                          rerank_sec=rerank_f_sec, expansion_requested=True,
                          expansion_applied=bool(set(weights) - set(tokenize(query))))
            artifact_rows[-1]["rm3_term_weights"] = weights

    if not metric_rows:
        raise ValueError("Nenhuma query disponível em experiment_cache.")
    metrics = pd.DataFrame(metric_rows)
    metric_names = [f"ndcg@{k}", f"mrr@{k}", f"recall@{k}", f"map@{k}", f"candidate_recall@{n}"]
    aggregations = {name: (name, "mean") for name in metric_names}
    aggregations.update({
        "num_queries": ("query_id", "size"), "expansion_rate": ("expansion_requested", "mean"),
        "valid_expansion_rate": ("expansion_applied", "mean"),
        "avg_estimated_uncached_sec": ("estimated_uncached_sec", "mean"),
        "avg_composed_observed_sec": ("composed_observed_sec", "mean"),
    })
    summary = metrics.groupby(["dataset", "pipeline_id", "pipeline"], as_index=False).agg(**aggregations)
    return {
        "metrics": metrics, "summary": summary, "rankings": pd.DataFrame(ranking_records),
        "decisions": pd.DataFrame(decision_rows), "artifacts": artifact_rows,
        "generations": expansion_cache.records,
        "metadata": {
            "config": dict(config), "datasets": dataset_records,
            "python": platform.python_version(),
            "packages": {name: package_version(name) for name in ("torch", "transformers", "rank-bm25", "numpy", "pandas", "scipy")},
            "reranker": config["reranker_model"],
            "reranker_revision": config["reranker_revision"],
            "reranker_max_length": config["reranker_max_length"], "reranker_batch_size": config["reranker_batch_size"],
            "reranker_setup_sec": getattr(rerank_fn, "setup_sec", 0.0),
            "llm_setup_sec": getattr(expansion_cache.generator, "setup_sec", 0.0),
            "llm_resolved_commit": getattr(getattr(getattr(expansion_cache.generator, "model", None), "config", None), "_commit_hash", None),
            "experiment_wall_sec": time.perf_counter() - experiment_start,
            "time_protocol": "Custos compostos por política; estimated_uncached_sec inclui geração histórica do cache. Setup separado. Não é latência end-to-end medida em produção.",
            "selector_protocol": "Regras fixas, sem qrels; nenhuma calibração/seleção de limiar nas queries de teste.",
            "rm3_protocol": "RM1 unigram de feedback com softmax(scores BM25), top termos normalizados, interpolação com query e BM25 ponderado; não é reprodução exata do Anserini.",
        },
    }
