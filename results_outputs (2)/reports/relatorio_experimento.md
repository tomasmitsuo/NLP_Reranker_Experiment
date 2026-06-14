# Relatório do experimento BEIR: BM25, BGE-Reranker e análises complementares

Gerado em: 2026-06-14 19:38:45

## Datasets
- **SciFact**: 5183 documentos, 300 queries, média de 1.13 docs relevantes por query.
- **TREC-COVID**: 171332 documentos, 50 queries, média de 493.46 docs relevantes por query.
- **NF-Corpus**: 3633 documentos, 323 queries, média de 38.19 docs relevantes por query.

## Resumo principal

| dataset    | pipeline            |   nDCG@10 |   MRR@10 |   Recall@10 |   MAP@10 |   avg_time_sec |   total_time_sec |   num_queries |   nDCG@5 |    MRR@5 |   Recall@5 |    MAP@5 |   nDCG@20 |   MRR@20 |   Recall@20 |   MAP@20 |   candidate_recall@100 |
|:-----------|:--------------------|----------:|---------:|------------:|---------:|---------------:|-----------------:|--------------:|---------:|---------:|-----------:|---------:|----------:|---------:|------------:|---------:|-----------------------:|
| NF-Corpus  | BM25                |  0.306317 | 0.508971 |   0.152153  | 0.219001 |     0.00426332 |          1.37705 |           323 | 0.334028 | 0.499742 | 0.11505    | 0.268051 |  0.278253 | 0.510282 |   0.172116  | 0.178807 |              0.235936  |
| NF-Corpus  | BM25 + BGE-Reranker |  0.30862  | 0.527299 |   0.142758  | 0.21934  |     3.04134    |        982.352   |           323 | 0.343844 | 0.520795 | 0.115398   | 0.274281 |  0.28011  | 0.529833 |   0.167174  | 0.178012 |              0.235936  |
| SciFact    | BM25                |  0.651893 | 0.618615 |   0.774     | 0.606975 |     0.024356   |          7.30681 |           300 | 0.636032 | 0.613222 | 0.728444   | 0.599106 |  0.667184 | 0.622329 |   0.832278  | 0.61146  |              0.873056  |
| SciFact    | BM25 + BGE-Reranker |  0.694917 | 0.669444 |   0.801833  | 0.654127 |     3.07088    |        921.264   |           300 | 0.676903 | 0.663    | 0.752611   | 0.645    |  0.706466 | 0.672174 |   0.845722  | 0.657581 |              0.873056  |
| TREC-COVID | BM25                |  0.576817 | 0.758    |   0.0152927 | 0.523202 |     0.983842   |         49.1921  |            50 | 0.60615  | 0.751333 | 0.00798443 | 0.573867 |  0.52454  | 0.759667 |   0.0273645 | 0.448882 |              0.0935702 |
| TREC-COVID | BM25 + BGE-Reranker |  0.643443 | 0.871524 |   0.0163412 | 0.605649 |     3.90638    |        195.319   |            50 | 0.691654 | 0.868667 | 0.00926469 | 0.686733 |  0.617824 | 0.871524 |   0.0316372 | 0.559027 |              0.0935702 |

## Conclusões automáticas

| dataset    |   delta_nDCG@10 |   delta_MRR@10 |   delta_Recall@10 |   bm25_avg_time_sec |   bge_avg_time_sec |   time_ratio_bge_vs_bm25 |   candidate_recall@100 |   wilcoxon_p_value_nDCG@10 | wilcoxon_significant_0_05   | conclusion   |
|:-----------|----------------:|---------------:|------------------:|--------------------:|-------------------:|-------------------------:|-----------------------:|---------------------------:|:----------------------------|:-------------|
| NF-Corpus  |      0.00230323 |      0.0183277 |       -0.00939507 |          0.00426332 |            3.04134 |                713.373   |              0.235936  |                0.950775    | False                       | melhorou     |
| SciFact    |      0.0430248  |      0.0508294 |        0.0278333  |          0.024356   |            3.07088 |                126.083   |              0.873056  |                0.000466468 | True                        | melhorou     |
| TREC-COVID |      0.066626   |      0.113524  |        0.00104847 |          0.983842   |            3.90638 |                  3.97053 |              0.0935702 |                0.00877843  | True                        | melhorou     |

## Diagnóstico de falhas

| dataset    | failure_type                                              |   num_queries |
|:-----------|:----------------------------------------------------------|--------------:|
| NF-Corpus  | falha_bm25_sem_relevantes_no_top100                       |            73 |
| NF-Corpus  | falha_ou_piora_do_reranker                                |            91 |
| NF-Corpus  | relevantes_existiam_nos_candidatos_mas_ficaram_fora_top10 |            16 |
| NF-Corpus  | sem_mudanca_relevante                                     |            56 |
| NF-Corpus  | sucesso_do_reranker                                       |            87 |
| SciFact    | falha_bm25_sem_relevantes_no_top100                       |            35 |
| SciFact    | falha_ou_piora_do_reranker                                |            29 |
| SciFact    | relevantes_existiam_nos_candidatos_mas_ficaram_fora_top10 |            13 |
| SciFact    | sem_mudanca_relevante                                     |           160 |
| SciFact    | sucesso_do_reranker                                       |            63 |
| TREC-COVID | falha_ou_piora_do_reranker                                |            17 |
| TREC-COVID | sem_mudanca_relevante                                     |             2 |
| TREC-COVID | sucesso_do_reranker                                       |            31 |

## Overlap BM25 vs BGE

| dataset    |   avg_top10_overlap |   avg_top10_overlap_ratio |
|:-----------|--------------------:|--------------------------:|
| NF-Corpus  |             3.85449 |                  0.385449 |
| SciFact    |             4.73333 |                  0.473333 |
| TREC-COVID |             2.22    |                  0.222    |

## RRF BM25 + BGE

| dataset    | pipeline       |   nDCG@10 |   MRR@10 |   Recall@10 |   MAP@10 |   avg_time_sec |   total_time_sec |   num_queries |   nDCG@5 |    MRR@5 |   Recall@5 |    MAP@5 |   nDCG@20 |   MRR@20 |   Recall@20 |   MAP@20 |
|:-----------|:---------------|----------:|---------:|------------:|---------:|---------------:|-----------------:|--------------:|---------:|---------:|-----------:|---------:|----------:|---------:|------------:|---------:|
| NF-Corpus  | RRF BM25 + BGE |  0.317711 | 0.545983 |    0.148118 | 0.227282 |              0 |                0 |           323 | 0.354947 | 0.540609 | 0.122924   | 0.285592 |  0.291427 | 0.547697 |    0.178349 | 0.186762 |
| SciFact    | RRF BM25 + BGE |  0.693752 | 0.659336 |    0.820778 | 0.646817 |              0 |                0 |           300 | 0.676209 | 0.653611 | 0.769556   | 0.638472 |  0.699363 | 0.660485 |    0.839611 | 0.648983 |
| TREC-COVID | RRF BM25 + BGE |  0.669321 | 0.925667 |    0.017417 | 0.6296   |              0 |                0 |            50 | 0.726466 | 0.925667 | 0.00971261 | 0.726333 |  0.604654 | 0.925667 |    0.030447 | 0.537903 |

## Figuras principais

- `metrics_mean_comparison.png`
- `metric_improvements_delta.png`
- `query_level_delta_distribution.png`
- `precision_recall_curves.png`
- `quality_cost_nDCG_at_10.png`
- `quality_cost_Recall_at_10.png`
- `failure_diagnosis_summary.png`
- `metrics_heatmap.png`
- `top10_overlap_bm25_bge.png`
- `relevant_documents_rank_movements.png`

## Arquivos de auditoria qualitativa

- `query_rankings_bm25_vs_bge_all_datasets.json`
- `qualitative_query_analysis_<dataset>.json`
- `qualitative_chunk_analysis_<dataset>.json`
- `qualitative_reranking_analysis_<dataset>.json`