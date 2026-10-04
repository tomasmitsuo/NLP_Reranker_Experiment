import math

def dcg_at_k(relevances, k):
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        position = i + 1
        dcg += rel / math.log2(position + 1)
    return dcg


def ndcg_at_k(retrieved_doc_ids, relevant_docs, k):
    relevances = [relevant_docs.get(doc_id, 0) for doc_id in retrieved_doc_ids[:k]]
    dcg = dcg_at_k(relevances, k)
    ideal_relevances = sorted(relevant_docs.values(), reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)
    return 0.0 if idcg == 0 else dcg / idcg


def mrr_at_k(retrieved_doc_ids, relevant_docs, k):
    for i, doc_id in enumerate(retrieved_doc_ids[:k]):
        if relevant_docs.get(doc_id, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(retrieved_doc_ids, relevant_docs, k):
    relevant_doc_ids = {
        doc_id for doc_id, relevance in relevant_docs.items() if relevance > 0
    }
    if not relevant_doc_ids:
        return 0.0
    retrieved_at_k = set(retrieved_doc_ids[:k])
    return len(retrieved_at_k.intersection(relevant_doc_ids)) / len(relevant_doc_ids)


def average_precision_at_k(retrieved_doc_ids, relevant_docs, k):
    relevant_doc_ids = {
        doc_id for doc_id, relevance in relevant_docs.items() if relevance > 0
    }
    if not relevant_doc_ids:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for i, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id in relevant_doc_ids:
            hits += 1
            precision_sum += hits / i

    return precision_sum / min(len(relevant_doc_ids), k)
