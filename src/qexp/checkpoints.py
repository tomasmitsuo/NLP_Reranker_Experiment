"""Retomada por query com invalidação por código, parâmetros e conteúdo dos dados."""
import hashlib
import io
import json
import time
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from .runner import run_qexp_experiment
from .selection import qexp_validate_config
from .progress import report
from .config import QEXP_CONFIG, pipeline_ids

TABLES = ("metrics", "rankings", "decisions")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save_result(result, path):
    from .naming import rename_table, rename_records, public_config
    payload = {key: value for key, value in result.items() if key not in (*TABLES, "summary")}
    public = result["metadata"]["config"].get("id_scheme") == "technique_selector"
    payload["tables"] = {name: (rename_table(result[name]) if public else result[name]).to_json(orient="table", index=False, double_precision=15) for name in TABLES}
    if public:
        payload["metadata"] = {**result["metadata"], "config": public_config(result["metadata"]["config"])}
        payload["artifacts"] = rename_records(result["artifacts"])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def load_result(path):
    from .naming import rename_table, rename_records, normalize_config
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    tables = result.pop("tables")
    result.update({name: pd.read_json(io.StringIO(value), orient="table") for name, value in tables.items()})
    for name in TABLES:
        result[name] = rename_table(result[name], public=False)
    result["artifacts"] = rename_records(result["artifacts"], public=False)
    result["metadata"]["config"] = normalize_config(result["metadata"]["config"])
    result["summary"] = summarize(result["metrics"], result["metadata"]["config"])
    return result


def summarize(metrics, config):
    k, n = config["eval_k"], config["candidate_n"]
    names = [f"ndcg@{k}", f"mrr@{k}", f"recall@{k}", f"map@{k}", f"candidate_recall@{n}"]
    aggregations = {name: (name, "mean") for name in names}
    aggregations.update(
        num_queries=("query_id", "size"), expansion_rate=("expansion_requested", "mean"),
        valid_expansion_rate=("expansion_applied", "mean"),
        avg_estimated_uncached_sec=("estimated_uncached_sec", "mean"),
        avg_composed_observed_sec=("composed_observed_sec", "mean"),
    )
    return metrics.groupby(["dataset", "pipeline_id", "pipeline"], as_index=False).agg(**aggregations)


def run_with_checkpoints(experiments, config, expansion_cache, rerank_fn, checkpoint_dir,
                         resume=True, checkpoint_tag="default"):
    """Uma query completa é a unidade de retomada; custos salvos são históricos.

    Ao injetar outro gerador/reranker, altere checkpoint_tag para identificá-lo.
    """
    from .naming import normalize_config
    config = normalize_config({**QEXP_CONFIG, **config})
    qexp_validate_config(config)
    selector = None
    if "L" in pipeline_ids(config):
        from .learned_selection import LearnedSelector
        selector = LearnedSelector.load(config["selector_model_path"])
        selector.validate_protocol(config)
    start = time.perf_counter()
    code_hash = digest({p.name: p.read_text() for p in sorted(Path(__file__).parent.glob("*.py"))})
    runtime = {name: version(name) for name in ("torch", "transformers", "rank-bm25", "numpy", "pandas", "scipy", "scikit-learn")}
    parameters = {key: value for key, value in config.items() if key not in {"cache_dir", "max_queries_per_dataset"}}
    if selector:
        parameters["selector_content_sha256"] = digest(selector.artifact)
    pieces, dataset_records = [], []
    reused = executed = 0
    limit = config["max_queries_per_dataset"]
    total = sum(len(list(case["queries"].items())[:limit]) for case in experiments.values())
    report("Preparando avaliação", total_queries=total)
    for slug, case in experiments.items():
        report("Verificando dados e checkpoints", dataset=case["config"]["name"], query_id="")
        # Hash único por corpus; não recalcula o corpus para cada query.
        corpus_key = digest({"documents": case["documents"], "doc_ids": case["doc_ids"],
                             "source": case["config"], "k1": case["bm25"].k1,
                             "b": case["bm25"].b, "epsilon": case["bm25"].epsilon})
        queries = list(case["queries"].items())[:config["max_queries_per_dataset"]]
        dataset_records.append({"dataset": case["config"]["name"], "source": case["config"],
                                "documents": len(case["documents"]), "query_ids": [str(q) for q, _ in queries],
                                "queries_sha256": digest(queries), "corpus_sha256": corpus_key,
                                "bm25_k1": case["bm25"].k1, "bm25_b": case["bm25"].b})
        for qid, query in queries:
            if selector:
                selector.check_unseen(case["config"]["name"], qid, query)
            report("Verificando checkpoint", query_id=str(qid))
            key = digest({"code": code_hash, "runtime": runtime, "config": parameters,
                          "corpus": corpus_key, "qid": str(qid), "query": query,
                          "qrels": case["qrels"].get(str(qid), {}), "tag": checkpoint_tag})
            path = Path(checkpoint_dir) / slug / f"{key}.json"
            if resume and path.exists():
                piece = load_result(path)
                reused += 1
                report("Checkpoint reutilizado", complete=True, reused=True)
            else:
                one_case = {**case, "queries": {qid: query}}
                record_start = len(expansion_cache.records)
                piece = run_qexp_experiment({slug: one_case}, config, expansion_cache, rerank_fn)
                piece["generations"] = expansion_cache.records[record_start:]
                report("Salvando checkpoint")
                save_result(piece, path)
                executed += 1
                report("Query concluída", complete=True)
            pieces.append(piece)
    if not pieces:
        raise ValueError("Nenhuma query disponível para execução.")
    result = {name: pd.concat([p[name] for p in pieces], ignore_index=True) for name in TABLES}
    for name in ("artifacts", "generations"):
        result[name] = [item for p in pieces for item in p[name]]
    result["metadata"] = {**pieces[0]["metadata"], "config": dict(config), "datasets": dataset_records,
                          "experiment_wall_sec": time.perf_counter() - start,
                          "reranker_setup_sec": getattr(rerank_fn, "setup_sec", 0.0),
                          "llm_setup_sec": getattr(expansion_cache.generator, "setup_sec", 0.0),
                          "checkpoint_queries_reused": reused, "checkpoint_queries_executed": executed,
                          "checkpoint_code_sha256": code_hash, "checkpoint_tag": checkpoint_tag,
                          "checkpoint_timing_note": "Tempos de queries retomadas são históricos; não representam latência desta execução."}
    result["summary"] = summarize(result["metrics"], config)
    return result
