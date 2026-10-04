# Experimento do TCC: mantenha estes parâmetros fixos antes de avaliar o teste.
QEXP_CONFIG = {
    "reranker_model": "BAAI/bge-reranker-base",
    "reranker_revision": "main",
    "reranker_batch_size": 16,
    "reranker_max_length": 512,
    "candidate_n": 100,
    "eval_k": 10,
    "max_queries_per_dataset": None,  # Use 3 para uma execução de verificação.
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "model_revision": "main",  # Fixe um commit do modelo para a execução final.
    "model_device": "auto",
    "seed": 42,
    "max_new_tokens": 512,
    "num_reformulations": 4,  # Reformulações adicionais; a query original sempre entra no RRF.
    "query_repetitions": 1,  # Query2doc simples; 5 permite estudar o peso lexical da query.
    "cache_dir": ".cache/query_expansion",
    "rrf_k": 60,
    "signal_top_k": 10,
    "gap_threshold": 0.15,  # Gap relativo: (score_1 - score_10) / abs(score_1).
    "entropy_threshold": 0.90,  # Entropia de softmax / log(n), entre 0 e 1.
    "entropy_temperature": 1.0,
    "overlap_threshold": 0.30,
    "selector_rule": "any",  # any, gap, entropy ou overlap; sem ajuste nos qrels de teste.
    "run_reranker_aware": True,  # E_RA: acrescenta divergência e contabiliza o BGE inicial.
    "divergence_threshold": 0.50,
    "run_h4": True,  # C_H4: mesmos candidatos de C, com query expandida no BGE.
    "rm3_fb_docs": 10,
    "rm3_fb_terms": 20,
    "rm3_original_weight": 0.50,
    "rm3_temperature": 1.0,
}

QEXP_PIPELINES = {
    "A": "BM25",
    "B": "BM25 + BGE-Reranker",
    "C": "Query2doc + BM25 + BGE",
    "D": "Multi-query + RRF + BGE",
    "E": "Selective Multi-query + RRF + BGE",
    "E_RA": "Selective Multi-query + RRF + BGE (reranker-aware)",
    "F": "RM3 + BM25 + BGE",
    "C_H4": "Query2doc + BM25 + BGE (query expandida)",
}

DATASETS = [
    {
        "name": "SciFact",
        "slug": "scifact",
        "docs_id": "beir/scifact",
        "eval_id": "beir/scifact/test",
    },
    {
        "name": "TREC-COVID",
        "slug": "trec_covid",
        "docs_id": "beir/trec-covid",
        "eval_id": "beir/trec-covid",
    },
    {
        "name": "NF-Corpus",
        "slug": "nfcorpus",
        "docs_id": "beir/nfcorpus",
        "eval_id": "beir/nfcorpus/test",
    },
]
