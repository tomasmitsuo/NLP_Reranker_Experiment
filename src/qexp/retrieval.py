import re
from collections import Counter
import numpy as np

def tokenize(text):
    return re.findall(r"\w+", text.lower())


def retrieve_bm25(query, bm25, doc_ids, top_k=10):
    scores = bm25.get_scores(tokenize(query))
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        {"doc_id": doc_ids[idx], "score": float(scores[idx])}
        for idx in top_indices
    ]

def qexp_softmax(scores, temperature=1.0):
    values = np.asarray(scores, dtype=float)
    if not len(values):
        return values
    if not np.isfinite(values).all():
        raise ValueError("Scores não finitos na recuperação.")
    weights = np.exp((values - values.max()) / temperature)
    return weights / weights.sum()


def qexp_rrf(rankings, rrf_k, top_k):
    scores = {}
    for ranking in rankings:
        seen = set()
        for rank, item in enumerate(ranking, start=1):
            doc_id = item["doc_id"]
            # Um documento recebe no máximo um voto de cada ranking.
            if doc_id not in seen:
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
                seen.add(doc_id)
    # sorted é estável: empate preserva a ordem da primeira aparição, começando em q0.
    return [{"doc_id": doc_id, "score": float(score)}
            for doc_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]]


def qexp_multi_candidates(query, expansions, initial, case, config):
    if not expansions:
        return initial
    rankings = [initial]
    seen = {query.casefold().strip()}
    for reformulation in expansions:
        if reformulation.casefold().strip() in seen:
            continue
        seen.add(reformulation.casefold().strip())
        rankings.append(retrieve_bm25(reformulation, case["bm25"], case["doc_ids"],
                                     top_k=config["candidate_n"]))
    return qexp_rrf(rankings, config["rrf_k"], config["candidate_n"])


def qexp_rm3_weights(query, initial, documents, config):
    """RM1 dos documentos de feedback, interpolado com o modelo unigram da query (RM3)."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    counts = Counter(tokenize(query))
    if not counts:
        return {}
    query_model = {term: count / sum(counts.values()) for term, count in counts.items()}
    feedback = initial[:config["rm3_fb_docs"]]
    posterior = qexp_softmax([item["score"] for item in feedback], config["rm3_temperature"])
    relevance_model = Counter()
    for item, probability in zip(feedback, posterior):
        terms = Counter(term for term in tokenize(documents[item["doc_id"]])
                        if len(term) > 1 and term not in ENGLISH_STOP_WORDS)
        length = sum(terms.values())
        if length:
            for term, count in terms.items():
                relevance_model[term] += float(probability) * count / length
    selected = dict(sorted(relevance_model.items(), key=lambda item: (-item[1], item[0]))
                    [:config["rm3_fb_terms"]])
    total = sum(selected.values())
    if not total:
        return query_model
    alpha = config["rm3_original_weight"]
    weights = {term: alpha * probability for term, probability in query_model.items()}
    for term, probability in selected.items():
        weights[term] = weights.get(term, 0.0) + (1.0 - alpha) * probability / total
    return {term: weight for term, weight in weights.items() if weight > 0}


def qexp_weighted_bm25(weights, bm25, doc_ids, top_k):
    # rank_bm25 recebe tokens sem pesos; somamos contribuições por termo para preservar
    # os pesos contínuos do RM3, sem aproximá-los pela repetição/arredondamento de tokens.
    scores = np.zeros(len(doc_ids), dtype=float)
    for term, weight in weights.items():
        scores += weight * bm25.get_scores([term])
    indices = np.argsort(scores)[::-1][:top_k]
    return [{"doc_id": doc_ids[index], "score": float(scores[index])} for index in indices]
