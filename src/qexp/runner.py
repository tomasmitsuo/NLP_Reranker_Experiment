import hashlib
import json
import platform
import time
from importlib.metadata import version as package_version
import pandas as pd
from tqdm.auto import tqdm
from .config import QEXP_PIPELINES, QEXP_CONFIG, pipeline_ids, CONTROLLED_METHODS
from .metrics import ndcg_at_k, mrr_at_k, recall_at_k, average_precision_at_k
from .retrieval import retrieve_bm25, tokenize, qexp_multi_candidates, qexp_rm3_weights, qexp_weighted_bm25
from .selection import qexp_validate_config, qexp_difficulty_signals, qexp_rank_divergence, qexp_should_expand
from .timing import qexp_timed
from .progress import report, active
from .selector_features import lexical_features, bge_features, CorpusStatistics, uses_bge, uses_rarity
from .feedback import bert_qe_rerank, keyword_rerank, controlled_rerank

def run_qexp_experiment(experiments, config, expansion_cache, rerank_fn):
    """Reavalia A–F sobre as mesmas queries; métricas antigas continuam independentes."""
    from .naming import normalize_config, public_id
    config = normalize_config({**QEXP_CONFIG, **config})
    qexp_validate_config(config)
    enabled = pipeline_ids(config)
    selector = None
    if "L" in enabled:
        from .learned_selection import LearnedSelector
        selector = LearnedSelector.load(config["selector_model_path"])
        selector.validate_protocol(config)
    metric_rows, ranking_records, decision_rows, artifact_rows, dataset_records = [], [], [], [], []
    experiment_start = time.perf_counter()
    n, k = config["candidate_n"], config["eval_k"]

    for case in experiments.values():
        dataset = case["config"]["name"]
        corpus_statistics = None
        if config["collect_query_statistics"]:
            if not hasattr(case["bm25"], "_qexp_corpus_statistics"):
                case["bm25"]._qexp_corpus_statistics = CorpusStatistics(case["bm25"])
            corpus_statistics = case["bm25"]._qexp_corpus_statistics
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
            if selector:
                selector.check_unseen(dataset, query_id, query)
            query_wall_start = time.perf_counter()
            shared_generations = {}

            def generation_for(kind):
                if kind not in shared_generations:
                    shared_generations[kind] = expansion_cache.get(query, kind)
                return shared_generations[kind]

            report("t1: recuperação BM25", dataset=dataset, query_id=query_id)
            initial, initial_sec = qexp_timed(retrieve_bm25, query, case["bm25"], case["doc_ids"], top_k=n)

            def record_result(pipeline_id, candidates, final, *, retrieval_sec, rerank_sec=0.0,
                              decision_sec=0.0, initial_rerank_sec=0.0, generation=None,
                              expansion_requested=False, expansion_applied=False, rerank_query=None, extra=None, policy_wall_sec=None):
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
                    "policy_wall_sec": policy_wall_sec,
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
                    **(extra or {}),
                })

            signals, signal_sec = qexp_timed(qexp_difficulty_signals, query, initial, case["documents"], config)
            features, lexical_sec = qexp_timed(lexical_features, query, signals)
            feature_sec = signal_sec + lexical_sec
            corpus_feature_sec = 0.
            if corpus_statistics:
                corpus_features, corpus_feature_sec = qexp_timed(corpus_statistics.features, query)
                features.update(corpus_features)
            bge_feature_sec = 0.0
            outcomes = {"A": dict(candidates=initial, final=initial, retrieval_sec=initial_sec)}

            def baseline():
                nonlocal bge_feature_sec
                if "B" not in outcomes:
                    report("t2: reranqueamento BGE original")
                    final, elapsed = qexp_timed(rerank_fn, query, initial, case["documents"])
                    extra_features, bge_feature_sec = qexp_timed(bge_features, initial, final, config["signal_top_k"])
                    features.update(extra_features)
                    outcomes["B"] = dict(candidates=initial, final=final, retrieval_sec=initial_sec,
                                         rerank_sec=elapsed, extra={"selector_features": dict(features),
                                         "feature_lexical_sec": feature_sec, "feature_bge_sec": bge_feature_sec,
                                         "feature_corpus_sec": corpus_feature_sec,
                                         "corpus_statistics_setup_sec": corpus_statistics.setup_sec if corpus_statistics else 0.})
                return outcomes["B"]

            def compute(method):
                if method in outcomes:
                    return outcomes[method]
                report(f"{public_id(method)}: recuperação e reranqueamento")
                if method == "B":
                    return baseline()
                if method in {"C", "C_H4"}:
                    if method == "C_H4":
                        source = compute("C")
                        expanded_query = source["extra"]["expanded_query"]
                        final, elapsed = qexp_timed(rerank_fn, expanded_query, source["candidates"], case["documents"])
                        outcome = {**source, "final": final, "rerank_sec": elapsed, "rerank_query": expanded_query}
                        outcome["initial_rerank_sec"] = source.get("initial_rerank_sec", 0.) + source["rerank_sec"]
                    else:
                        generation = expansion_cache.get(query, "query2doc")
                        expanded_query = " ".join([query] * config["query_repetitions"] + generation["expansions"])
                        if generation["expansions"]:
                            candidates, elapsed = qexp_timed(retrieve_bm25, expanded_query, case["bm25"], case["doc_ids"], top_k=n)
                            final, rerank_sec = qexp_timed(rerank_fn, query, candidates, case["documents"])
                            outcome = dict(candidates=candidates, final=final, retrieval_sec=elapsed, rerank_sec=rerank_sec)
                        else:
                            outcome = dict(baseline())
                            expanded_query = query
                        outcome.update(generation=generation, expansion_requested=True,
                                       expansion_applied=bool(generation["expansions"]), extra={"expanded_query": expanded_query})
                elif method == "D":
                    generation = expansion_cache.get(query, "multi")
                    if generation["expansions"]:
                        candidates, elapsed = qexp_timed(qexp_multi_candidates, query, generation["expansions"], initial, case, config)
                        final, rerank_sec = qexp_timed(rerank_fn, query, candidates, case["documents"])
                        outcome = dict(candidates=candidates, final=final, retrieval_sec=initial_sec + elapsed, rerank_sec=rerank_sec)
                    else:
                        outcome = dict(baseline())
                    outcome.update(generation=generation, expansion_requested=True,
                                   expansion_applied=bool(generation["expansions"]))
                elif method == "F":
                    weights, feedback_sec = qexp_timed(qexp_rm3_weights, query, initial, case["documents"], config)
                    candidates, elapsed = qexp_timed(qexp_weighted_bm25, weights, case["bm25"], case["doc_ids"], n)
                    final, rerank_sec = qexp_timed(rerank_fn, query, candidates, case["documents"])
                    outcome = dict(candidates=candidates, final=final, retrieval_sec=initial_sec + feedback_sec + elapsed,
                                   rerank_sec=rerank_sec, expansion_requested=True,
                                   expansion_applied=bool(set(weights) - set(tokenize(query))), extra={"rm3_term_weights": weights})
                elif method in {"G", "K"}:
                    base = baseline()
                    if method == "G":
                        (final, chunks), elapsed = qexp_timed(bert_qe_rerank, query, initial, base["final"], case["documents"], config, rerank_fn)
                        outcome = dict(final=final, rerank_sec=elapsed, expansion_applied=bool(chunks), extra={"feedback_chunks": chunks})
                    else:
                        generation = generation_for("keywords")
                        if generation["expansions"]:
                            (final, expanded_queries), elapsed = qexp_timed(keyword_rerank, query, generation["expansions"], initial,
                                                                           base["final"], case["documents"], config, rerank_fn)
                        else:
                            final, expanded_queries, elapsed = base["final"], [], 0.0
                        outcome = dict(final=final, rerank_sec=elapsed, generation=generation,
                                       expansion_applied=bool(expanded_queries), extra={"reranker_queries": [query, *expanded_queries]})
                    outcome.update(candidates=initial, retrieval_sec=initial_sec, initial_rerank_sec=base["rerank_sec"], expansion_requested=True)
                elif method in CONTROLLED_METHODS:
                    base = baseline()
                    generation = generation_for("keywords")
                    expansion = "; ".join(generation["expansions"])
                    if expansion:
                        (final, input_query), elapsed = qexp_timed(
                            controlled_rerank, query, expansion, initial, base["final"],
                            case["documents"], config, rerank_fn, method)
                    else:
                        final, input_query, elapsed = base["final"], query, 0.0
                    outcome = dict(candidates=initial, final=final, retrieval_sec=initial_sec,
                                   initial_rerank_sec=base["rerank_sec"], rerank_sec=elapsed,
                                   generation=generation, expansion_requested=True,
                                   expansion_applied=bool(expansion), rerank_query=input_query,
                                   extra={"controlled_expansion": expansion, "incorporation": method,
                                          "reranker_queries": [query, input_query]})
                elif method in {"E", "E_RA", "L", "R"}:
                    needs_bge = method == "E_RA" or (method == "L" and uses_bge(selector.artifact["feature_set"]))
                    if needs_bge or config["selector_baseline_first"]:
                        baseline()
                    probability = None
                    benefit_score = None
                    target = config["heuristic_method"] if method in {"E", "E_RA"} else config["selective_method"]
                    start = time.perf_counter()
                    if method == "L":
                        target, benefit_score = selector.choose(features)
                        expand = target != "B"
                        if selector.artifact["schema"] == 1:
                            probability = benefit_score
                        reason = ("learned_utility" if expand else "utility_fallback") if selector.artifact["schema"] == 3 else ("learned_benefit" if expand else "learned_no_benefit")
                        if selector.artifact["schema"] == 4:
                            reason = "learned_neural_utility" if expand else "neural_utility_fallback"
                    elif method == "R":
                        # Determinístico por query; independente de ordem, qrels e resultados expandidos.
                        value = hashlib.sha256(json.dumps([config["seed"], dataset, query_id]).encode()).hexdigest()
                        expand = int(value[:16], 16) / 2**64 < config["random_expansion_rate"]
                        reason = "random"
                    else:
                        expand, reason = qexp_should_expand(signals, config, features.get("rank_divergence") if method == "E_RA" else None)
                    decision_sec = time.perf_counter() - start
                    if method != "R":
                        decision_sec += feature_sec + (bge_feature_sec if needs_bge else 0.0)
                        if method == "L" and uses_rarity(selector.artifact["feature_set"]):
                            decision_sec += corpus_feature_sec
                    decision_rows.append({"dataset": dataset, "query_id": query_id, "pipeline_id": method,
                                          **signals, "rank_divergence": features.get("rank_divergence") if needs_bge else None,
                                          "expand": expand and target not in {"A", "B"}, "reason": reason, "selected_method": target if expand else "B",
                                          "benefit_probability": probability, "benefit_score": benefit_score})
                    # Só materializa a expansão depois da decisão; referências podem solicitá-la separadamente.
                    outcome = dict(compute(target) if expand else baseline())
                    if config["selector_baseline_first"] and expand:
                        base = baseline()
                        if target in {"C", "C_H4"} and outcome.get("expansion_applied"):
                            outcome["retrieval_sec"] += initial_sec
                        if target in {"A", "C", "C_H4", "D", "F"} and outcome["final"] is not base["final"]:
                            outcome["initial_rerank_sec"] = outcome.get("initial_rerank_sec", 0) + base["rerank_sec"]
                    outcome["extra"] = {**outcome.get("extra", {}), "selected_method": target if expand else "B"}
                    outcome["decision_sec"] = decision_sec
                    if not config["selector_baseline_first"] and method != "R" and target == "C" and expand and outcome.get("expansion_applied"):
                        outcome["retrieval_sec"] += initial_sec  # Features dependem da busca original.
                    if needs_bge and not outcome.get("initial_rerank_sec", 0):
                        base = baseline()
                        outcome["initial_rerank_sec"] = base["rerank_sec"]
                        if outcome["final"] is base["final"]:
                            outcome["rerank_sec"] = 0.0
                    outcome["expansion_requested"] = expand and target not in {"A", "B"}
                else:
                    raise ValueError(f"Pipeline desconhecido: {method}")
                outcomes[method] = outcome
                return outcome

            # As políticas decidem antes das referências de expansão incondicional.
            ordered = [m for m in ["A", "B", "L", "E", "E_RA", "R", "C", "C_H4", "D", "F", "G", "K", *CONTROLLED_METHODS] if m in enabled]
            for method in ordered:
                outcome = compute(method)
                if method == "L" or (config["selector_baseline_first"] and method in {"E", "E_RA", "R"}):
                    outcome = {**outcome, "policy_wall_sec": time.perf_counter() - query_wall_start}
                record_result(method, **outcome)

    if not metric_rows:
        raise ValueError("Nenhuma query disponível em experiment_cache.")
    metrics = pd.DataFrame(metric_rows)
    metrics["policy_wall_sec"] = pd.to_numeric(metrics["policy_wall_sec"], errors="raise").astype(float)
    metric_names = [f"ndcg@{k}", f"mrr@{k}", f"recall@{k}", f"map@{k}", f"candidate_recall@{n}"]
    aggregations = {name: (name, "mean") for name in metric_names}
    aggregations.update({
        "num_queries": ("query_id", "size"), "expansion_rate": ("expansion_requested", "mean"),
        "valid_expansion_rate": ("expansion_applied", "mean"),
        "avg_estimated_uncached_sec": ("estimated_uncached_sec", "mean"),
        "avg_composed_observed_sec": ("composed_observed_sec", "mean"),
    })
    summary = metrics.groupby(["dataset", "pipeline_id", "pipeline"], as_index=False).agg(**aggregations)
    decisions = pd.DataFrame(decision_rows)
    for column in ("benefit_probability", "benefit_score", "rank_divergence"):
        if column in decisions:
            decisions[column] = pd.to_numeric(decisions[column], errors="raise").astype(float)
    return {
        "metrics": metrics, "summary": summary, "rankings": pd.DataFrame(ranking_records),
        "decisions": decisions, "artifacts": artifact_rows,
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
            "selector_protocol": "E/E_RA: regras fixas. L: modelo e limiar ajustados em treino/validação externos; BGE em ambos os caminhos.",
            "enabled_pipelines": enabled,
            "selector_model": selector.artifact if selector else None,
            "policy_wall_protocol": "L: tempo observado desde início da consulta até a política, incluindo A/B, atributos, decisão e bookkeeping anterior. Pode incluir setup/cache; medir com sessão nova e registrar condição. Não soma execução de alternativas posteriores.",
            "feedback_protocol": "G: adaptação BERT-QE sem fine-tuning. K: keywords individuais e RRF dos rankings BGE. Candidatos fixos de B.",
            "rm3_protocol": "RM1 unigram de feedback com softmax(scores BM25), top termos normalizados, interpolação com query e BM25 ponderado; não é reprodução exata do Anserini.",
        },
    }
