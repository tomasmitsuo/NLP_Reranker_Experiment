import numpy as np
from .retrieval import tokenize, qexp_softmax

def qexp_validate_config(config):
    from .config import QEXP_PIPELINES, CONTROLLED_METHODS, ALL_TECHNIQUES
    from .naming import normalize_config
    config = normalize_config(config)
    if (config.get("id_scheme", "technique_selector") not in {"legacy", "technique_selector"}
            or config.get("heuristic_method", "D") not in ALL_TECHNIQUES
            or type(config.get("selector_baseline_first", False)) is not bool):
        raise ValueError("Nomenclatura ou protocolo seletivo inválido.")
    for key in ("candidate_n", "eval_k", "signal_top_k", "rm3_fb_docs", "rm3_fb_terms",
                "max_new_tokens", "query_repetitions", "num_reformulations"):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 1:
            raise ValueError(f"{key} deve ser um inteiro positivo.")
    if config["candidate_n"] < config["eval_k"]:
        raise ValueError("candidate_n deve ser pelo menos eval_k.")
    if not 3 <= config["num_reformulations"] <= 5:
        raise ValueError("Use de 3 a 5 reformulações.")
    if config["rrf_k"] <= 0 or config["entropy_temperature"] <= 0 or config["rm3_temperature"] <= 0:
        raise ValueError("Constante RRF e temperaturas devem ser positivas.")
    for key in ("gap_threshold", "entropy_threshold", "overlap_threshold",
                "divergence_threshold", "rm3_original_weight"):
        if not 0 <= config[key] <= 1:
            raise ValueError(f"{key} deve estar entre 0 e 1.")
    if config["selector_rule"] not in {"any", "gap", "entropy", "overlap"}:
        raise ValueError("Regra seletiva desconhecida.")
    if type(config["collect_query_statistics"]) is not bool:
        raise ValueError("collect_query_statistics deve ser booleano.")
    limit = config["max_queries_per_dataset"]
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        raise ValueError("max_queries_per_dataset deve ser None ou inteiro positivo.")
    pipelines = config.get("pipelines")
    if pipelines is not None and (not isinstance(pipelines, list) or not pipelines
                                  or any(p not in QEXP_PIPELINES for p in pipelines)):
        raise ValueError("pipelines deve ser uma lista não vazia de IDs conhecidos.")
    for key in ("keyword_count", "keyword_max_words", "bert_qe_docs", "bert_qe_chunk_words",
                "bert_qe_stride", "bert_qe_chunks", "bert_qe_max_chunks_per_doc"):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 1:
            raise ValueError(f"{key} deve ser inteiro positivo.")
    if config["bert_qe_stride"] > config["bert_qe_chunk_words"]:
        raise ValueError("bert_qe_stride não pode exceder bert_qe_chunk_words.")
    for key in ("bert_qe_alpha", "random_expansion_rate", "controlled_alpha"):
        if not 0 <= config[key] <= 1:
            raise ValueError(f"{key} deve estar entre 0 e 1.")
    if config["selective_method"] not in ALL_TECHNIQUES | {"portfolio"}:
        raise ValueError("selective_method deve ser C, D, F, G, K ou portfolio.")
    if config["selective_method"] == "portfolio" and (not pipelines or "R" in pipelines):
        raise ValueError("Portfolio exige pipelines explícitos e usa controle aleatório por replay, sem R.")
    if pipelines and "L" in pipelines and not config["selector_model_path"]:
        raise ValueError("L exige selector_model_path treinado em dados separados.")
    if config["generation_format"] not in {"strict_json", "numbered"}:
        raise ValueError("generation_format deve ser strict_json ou numbered.")
    if config["keyword_prompt_version"] not in {1, 2}:
        raise ValueError("keyword_prompt_version deve ser 1 ou 2.")
    if type(config["keyword_retries"]) is not int or not 0 <= config["keyword_retries"] <= 1:
        raise ValueError("keyword_retries deve ser 0 ou 1.")
    bounds = config["query2doc_words"]
    if (not isinstance(bounds, list) or len(bounds) != 2
            or any(not isinstance(v, int) or isinstance(v, bool) for v in bounds)
            or not 1 <= bounds[0] <= bounds[1]):
        raise ValueError("query2doc_words exige [mínimo, máximo] positivos.")
    examples = config["query2doc_examples"]
    if not isinstance(examples, list) or any(
        not isinstance(e, dict) or set(e) != {"query", "passage"}
        or any(not isinstance(v, str) or not v.strip() for v in e.values()) for e in examples
    ):
        raise ValueError("query2doc_examples exige pares query/passage não vazios.")


def qexp_difficulty_signals(query, candidates, documents, config):
    """Apenas query, scores e textos recuperados; não recebe qrels nem métricas."""
    top = candidates[:config["signal_top_k"]]
    if not top:
        return {"score_gap": 0.0, "relative_gap": 0.0, "score_entropy": 1.0,
                "lexical_overlap": 0.0, "signal_docs": 0}
    scores = [item["score"] for item in top]
    gap = float(scores[0] - scores[-1])
    probabilities = qexp_softmax(scores, config["entropy_temperature"])
    positive = probabilities[probabilities > 0]
    entropy = float(-(positive * np.log(positive)).sum() / np.log(len(top))) if len(top) > 1 else 0.0
    query_terms = set(tokenize(query))
    overlaps = [len(query_terms & set(tokenize(documents[item["doc_id"]]))) / max(1, len(query_terms))
                for item in top]
    return {
        "score_gap": gap,
        "relative_gap": gap / max(abs(scores[0]), 1e-12),
        "score_entropy": float(np.clip(entropy, 0.0, 1.0)),
        "lexical_overlap": float(np.mean(overlaps)),
        "signal_docs": len(top),
    }


def qexp_rank_divergence(initial, reranked, top_k=10):
    # Média do desacordo dos prefixos: detecta tanto substituições quanto mudanças de ordem.
    # Top-10 overlap isolado não detectaria a inversão dos mesmos dez documentos.
    depth = min(top_k, len(initial), len(reranked))
    if not depth:
        return 0.0
    left, right, agreements = set(), set(), []
    for i in range(depth):
        left.add(initial[i]["doc_id"])
        right.add(reranked[i]["doc_id"])
        agreements.append(len(left & right) / (i + 1))
    return 1.0 - float(np.mean(agreements))


def qexp_should_expand(signals, config, divergence=None):
    conditions = {
        "gap": signals["relative_gap"] < config["gap_threshold"],
        "entropy": signals["score_entropy"] > config["entropy_threshold"],
        "overlap": signals["lexical_overlap"] < config["overlap_threshold"],
    }
    rule = config["selector_rule"]
    reasons = [name for name, active in conditions.items() if active and rule in {"any", name}]
    if divergence is not None and divergence > config["divergence_threshold"]:
        reasons.append("bm25_bge_divergence")
    return bool(reasons), ";".join(reasons) if reasons else "easy"
