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
    "collect_query_statistics": False,  # DF/IDF do corpus; habilite para testar raridade.
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
    # None preserva as oito variantes históricas; listas explícitas incluem A/B automaticamente.
    "pipelines": None,
    "generation_format": "strict_json",  # numbered aceita também listas numeradas completas.
    "query2doc_examples": [],  # Pares query/passage definidos apenas em desenvolvimento.
    "query2doc_words": [80, 120],
    "keyword_prompt_version": 1,
    "keyword_retries": 0,  # v2 permite uma nova tentativa, sempre contabilizada.
    "controlled_alpha": 0.5,
    "keyword_count": 3,
    "keyword_max_words": 3,
    "bert_qe_docs": 3,
    "bert_qe_chunk_words": 32,
    "bert_qe_stride": 16,
    "bert_qe_chunks": 3,
    "bert_qe_max_chunks_per_doc": 32,
    "bert_qe_alpha": 0.5,
    "selective_method": "D",  # C/D/F/G/K ou portfolio (L schema 2); BGE em ambos os caminhos.
    "selector_model_path": None,
    "random_expansion_rate": 0.5,
    "id_scheme": "technique_selector",  # IDs públicos t1–t8/s1–s4; leitura histórica preservada.
    "heuristic_method": "D",
    "selector_baseline_first": False,  # Matriz completa cobra BGE original antes de qualquer decisão.
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
    "G": "Feedback por trechos + BGE (adaptação BERT-QE)",
    "K": "Expansões curtas + fusão de rankings BGE",
    "Q_CONCAT": "BGE: concatenação controlada",
    "Q_REPLACE": "BGE: substituição controlada",
    "Q_SCORE": "BGE: combinação controlada de escores",
    "Q_RRF": "BGE: fusão controlada de rankings",
    "L": "Expansão seletiva aprendida + BGE",
    "R": "Expansão aleatória + BGE",
}


CONTROLLED_METHODS = ("Q_CONCAT", "Q_REPLACE", "Q_SCORE", "Q_RRF")
FIXED_METHODS = {"G", "K", *CONTROLLED_METHODS}
ALL_TECHNIQUES = {"A", "B", "C", "C_H4", "D", "F", *FIXED_METHODS}

def pipeline_ids(config):
    from .naming import normalize_config
    config = normalize_config(config)
    if config.get("pipelines") is not None:
        return list(dict.fromkeys(["A", "B", *config["pipelines"]]))
    return ["A", "B", "C", "D", "E", "F"] + (
        ["E_RA"] if config["run_reranker_aware"] else []
    ) + (["C_H4"] if config["run_h4"] else [])

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
