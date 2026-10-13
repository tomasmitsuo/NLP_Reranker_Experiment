"""Intervenções no reranqueamento com candidatos fixos, sem acesso a qrels."""
import numpy as np

from .retrieval import qexp_softmax, qexp_rrf


def bert_qe_rerank(query, candidates, initial_ranking, documents, config, rerank_fn):
    """Adaptação das três fases de BERT-QE usando BGE já treinado.

    O ranking inicial é fornecido pelo chamador e cobrado separadamente.
    Os trechos são limitados por documento para tornar explícito o orçamento.
    """
    texts, origins = {}, {}
    for item in initial_ranking[:config["bert_qe_docs"]]:
        words = documents[item["doc_id"]].split()
        for index, start in enumerate(range(0, len(words), config["bert_qe_stride"])):
            if index >= config["bert_qe_max_chunks_per_doc"]:
                break
            key = f"chunk_{len(texts)}"
            texts[key] = " ".join(words[start:start + config["bert_qe_chunk_words"]])
            origins[key] = {"source_doc_id": item["doc_id"], "start_word": start}
            if start + config["bert_qe_chunk_words"] >= len(words):
                break
    if not texts:
        return initial_ranking, []
    scored = rerank_fn(query, [{"doc_id": key, "score": 0.0} for key in texts], texts)
    selected = scored[:config["bert_qe_chunks"]]
    weights = qexp_softmax([item["score"] for item in selected])
    feedback = {item["doc_id"]: 0.0 for item in candidates}
    artifacts = []
    for item, weight in zip(selected, weights):
        text = texts[item["doc_id"]]
        ranking = rerank_fn(text, candidates, documents)
        for document in ranking:
            feedback[document["doc_id"]] += float(weight) * document["score"]
        artifacts.append({**origins[item["doc_id"]], "text": text,
                          "query_score": float(item["score"]), "weight": float(weight)})
    alpha = config["bert_qe_alpha"]
    final = [{"doc_id": item["doc_id"],
              "score": float((1 - alpha) * item["score"] + alpha * feedback[item["doc_id"]])}
             for item in initial_ranking]
    if any(not np.isfinite(item["score"]) for item in final):
        raise ValueError("Score não finito no feedback BERT-QE.")
    return sorted(final, key=lambda item: item["score"], reverse=True), artifacts


def keyword_rerank(query, keywords, candidates, initial_ranking, documents, config, rerank_fn):
    """Uma expansão curta por inferência; RRF inclui o ranking original.

    Controle simples inspirado em expansão mínima; não reproduz o método de Li et al.
    """
    queries = [f"{query} {keyword}" for keyword in keywords]
    rankings = [initial_ranking] + [rerank_fn(text, candidates, documents) for text in queries]
    return qexp_rrf(rankings, config["rrf_k"], len(candidates)), queries


def controlled_rerank(query, expansion, candidates, original, documents, config, rerank_fn, method):
    """Uma expansão idêntica e candidatos fixos; varia apenas a incorporação.

    Combinação linear usa escores BGE brutos (mesmo modelo); RRF usa posições.
    Nenhuma normalização dependente de relevância ou ajuste por consulta.
    """
    from .config import CONTROLLED_METHODS
    if method not in CONTROLLED_METHODS:
        raise ValueError("Incorporação controlada desconhecida.")
    text = f"{query} {expansion}" if method == "Q_CONCAT" else expansion
    expanded = rerank_fn(text, candidates, documents)
    ids = {d["doc_id"] for d in candidates}
    for ranking in (original, expanded):
        if len(ranking) != len(ids) or {d["doc_id"] for d in ranking} != ids:
            raise ValueError("O reranqueador alterou os candidatos fixos.")
        if any(not np.isfinite(d["score"]) for d in ranking):
            raise ValueError("Score não finito.")
    if method == "Q_SCORE":
        extra = {d["doc_id"]: d["score"] for d in expanded}
        alpha = config["controlled_alpha"]
        final = [{"doc_id": d["doc_id"], "score": (1-alpha)*d["score"] + alpha*extra[d["doc_id"]]}
                 for d in original]
        final = sorted(final, key=lambda d: d["score"], reverse=True)
    elif method == "Q_RRF":
        final = qexp_rrf([original, expanded], config["rrf_k"], len(candidates))
    else:
        final = expanded
    return final, text
