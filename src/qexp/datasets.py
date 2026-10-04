"""Preparação independente dos dados: não executa baselines nem carrega BGE."""
import hashlib
import inspect
import json
import pickle
from importlib.metadata import version
from pathlib import Path

from rank_bm25 import BM25Okapi

from .retrieval import tokenize
from .progress import report


def load_beir_dataset(config, max_queries=None):
    import ir_datasets

    docs = ir_datasets.load(config["docs_id"])
    evaluation = ir_datasets.load(config["eval_id"])
    report("Lendo documentos (download quando necessário)")
    documents = {
        str(doc.doc_id): f"{getattr(doc, 'title', '') or ''}. {getattr(doc, 'text', '') or ''}".strip()
        for doc in docs.docs_iter()
    }
    report("Lendo queries e julgamentos")
    queries = {}
    for query in evaluation.queries_iter():
        queries[str(query.query_id)] = query.text
        if max_queries is not None and len(queries) >= max_queries:
            break
    qrels = {}
    for item in evaluation.qrels_iter():
        if str(item.query_id) in queries:
            qrels.setdefault(str(item.query_id), {})[str(item.doc_id)] = int(item.relevance)
    queries = {qid: query for qid, query in queries.items() if qid in qrels}
    return documents, queries, qrels


def prepare_dataset(config, cache_dir, max_queries=None):
    """Cache local confiável de corpus/índice, separado por split e limite de queries."""
    if max_queries is not None and (isinstance(max_queries, bool) or max_queries < 1):
        raise ValueError("max_queries deve ser positivo ou None")
    specification = {
        "dataset": config, "max_queries": max_queries,
        "loader": inspect.getsource(load_beir_dataset),
        "tokenizer": inspect.getsource(tokenize),
        "rank_bm25": version("rank-bm25"), "ir_datasets": version("ir-datasets"),
        "k1": 1.5, "b": 0.75, "epsilon": 0.25,
    }
    key = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    path = Path(cache_dir) / f"{config['slug']}-{key}.pkl"
    if path.exists():
        report("Carregando corpus/índice do cache")
        with path.open("rb") as stream:
            return pickle.load(stream)
    documents, queries, qrels = load_beir_dataset(config, max_queries)
    doc_ids = list(documents)
    report(f"Tokenizando {len(doc_ids)} documentos e construindo BM25")
    case = dict(config=config, documents=documents, queries=queries, qrels=qrels,
                doc_ids=doc_ids, bm25=BM25Okapi([tokenize(documents[d]) for d in doc_ids]))
    report("Salvando corpus/índice no cache")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(case, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    return case
