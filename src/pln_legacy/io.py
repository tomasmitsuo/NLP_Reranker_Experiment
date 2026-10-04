"""Lê os artefatos necessários ao experimento de reranker seletivo."""
from pathlib import Path
import pandas as pd


def load_baseline_results(directory):
    directory = Path(directory)

    def read(name):
        matches = list(directory.rglob(name))
        if len(matches) != 1:
            raise ValueError(f"Esperado um único {name} em {directory}; encontrados {len(matches)}. Escolha uma execução específica.")
        return pd.read_csv(matches[0], dtype={"query_id": str})

    rankings = read("rankings_all_datasets.csv")
    metrics = read("results_all_datasets.csv")
    required = {"BM25", "BM25 + BGE-Reranker"}
    if not required.issubset(set(metrics["pipeline"])):
        raise ValueError("Os resultados precisam conter BM25 e BM25 + BGE-Reranker.")
    if "doc_text" not in rankings:
        raise ValueError("O ranking precisa conter doc_text para as features lexicais.")
    keys = ["dataset", "query_id"]
    filtered = metrics[metrics.pipeline.isin(required)]
    if filtered.duplicated([*keys, "pipeline"]).any():
        raise ValueError("Métricas duplicadas para uma query/pipeline.")
    query_sets = [set(map(tuple, filtered[filtered.pipeline == method][keys].to_numpy())) for method in sorted(required)]
    if query_sets[0] != query_sets[1]:
        raise ValueError("BM25 e BGE precisam ter as mesmas queries.")
    bm25_queries = set(map(tuple, rankings[rankings.pipeline == "BM25"][keys].drop_duplicates().to_numpy()))
    if bm25_queries != query_sets[0]:
        raise ValueError("Rankings BM25 e métricas precisam ter as mesmas queries.")
    return rankings, metrics
