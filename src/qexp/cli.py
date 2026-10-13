"""Etapas independentes: prepare, generate, run e analyze."""
import argparse
import json
from pathlib import Path
from .config import DATASETS, QEXP_CONFIG, pipeline_ids, CONTROLLED_METHODS
from .progress import ProgressReporter, report


def load_settings(path):
    path = Path(path).resolve()
    settings = json.loads(path.read_text(encoding="utf-8"))
    unknown = set(settings.get("qexp", {})) - set(QEXP_CONFIG)
    if unknown:
        raise ValueError(f"Parâmetros desconhecidos: {sorted(unknown)}")
    settings["qexp"] = {**QEXP_CONFIG, **settings.get("qexp", {})}
    from .naming import normalize_config
    settings["qexp"] = normalize_config(settings["qexp"])
    root = (path.parent / settings.get("project_root", "..")).resolve()
    for key, default in (("output_dir", "results/query_expansion"),
                         ("prepared_dir", ".cache/prepared"),
                         ("checkpoint_dir", ".cache/qexp_checkpoints")):
        settings[key] = str(root / settings.get(key, default))
    settings["qexp"]["cache_dir"] = str(root / settings["qexp"]["cache_dir"])
    if settings["qexp"]["selector_model_path"]:
        settings["qexp"]["selector_model_path"] = str(root / settings["qexp"]["selector_model_path"])
    if settings.get("query_split_file"):
        import pandas as pd
        splits = pd.read_csv(root / settings["query_split_file"], dtype=str, keep_default_na=False)
        split = settings.get("query_split", "test")
        if (not {"dataset", "query_id", "split"} <= set(splits)
                or splits.duplicated(["dataset", "query_id"]).any()
                or split not in {"train", "validation", "test"}):
            raise ValueError("Manifesto/partição de consultas inválido.")
        selected = splits[splits.split == split]
        if selected.empty:
            raise ValueError("Partição de consultas vazia.")
        settings["query_filter"] = selected[["dataset", "query_id"]].values.tolist()
    return settings


def execute_stage(stage, settings, resume=True, progress="auto"):
    with ProgressReporter(settings["output_dir"], mode=progress):
        return _execute_stage(stage, settings, resume)


def _execute_stage(stage, settings, resume):
    output = Path(settings["output_dir"])
    if stage == "analyze":
        from .checkpoints import load_result
        from .evaluation import qexp_export
        report("Carregando resultados salvos")
        result = load_result(output / "qexp_run.json")
        report("Calculando hipóteses e exportando tabelas")
        return qexp_export(result, result["metadata"]["config"], output)

    import torch
    from .datasets import prepare_dataset
    from .selection import qexp_validate_config
    config = dict(settings["qexp"])
    if config["model_device"] == "auto":
        config["model_device"] = "cuda" if torch.cuda.is_available() else "cpu"
    qexp_validate_config(config)
    selected = settings.get("datasets", [entry["slug"] for entry in DATASETS])
    if not selected or set(selected) - {entry["slug"] for entry in DATASETS}:
        raise ValueError(f"Datasets inválidos: {selected}")
    experiments = {}
    report("Preparando datasets")
    for entry in DATASETS:
        if entry["slug"] in selected:
            report("Preparando corpus e índice BM25", dataset=entry["name"])
            experiments[entry["slug"]] = prepare_dataset(
                entry, settings["prepared_dir"], config["max_queries_per_dataset"]
            )
    if settings.get("query_filter") is not None:
        requested = {tuple(pair) for pair in settings["query_filter"]}
        available = {(case["config"]["name"], str(qid)) for case in experiments.values() for qid in case["queries"]}
        if not requested or not requested <= available:
            raise ValueError("Filtro contém consultas ausentes; confira datasets e limite de queries.")
        experiments = {slug: {**case, "queries": {qid: text for qid, text in case["queries"].items()
                       if (case["config"]["name"], str(qid)) in requested}}
                       for slug, case in experiments.items()}
    if stage == "prepare":
        return {slug: {"documents": len(case["documents"]), "queries": len(case["queries"])}
                for slug, case in experiments.items()}
    from .generation import QExpExpansionCache
    cache = QExpExpansionCache(config)
    if stage == "generate":
        methods = set(pipeline_ids(config))
        if methods & {"E", "E_RA"}:
            methods.add(config["heuristic_method"])
        if methods & {"L", "R"}:
            if config["selective_method"] == "portfolio":
                from .learned_selection import LearnedSelector
                selector = LearnedSelector.load(config["selector_model_path"])
                selector.validate_protocol(config)
                methods.update(selector.artifact["methods"])
            else:
                methods.add(config["selective_method"])
        report("Gerando expansões", total_queries=sum(len(case["queries"]) for case in experiments.values()))
        for case in experiments.values():
            for qid, query in case["queries"].items():
                report("Iniciando query", dataset=case["config"]["name"], query_id=str(qid))
                generated = []
                kinds = (["query2doc"] if methods & {"C", "C_H4"} else []) + (
                    ["multi"] if "D" in methods else []) + (["keywords"] if methods & {"K", *CONTROLLED_METHODS} else [])
                for kind in kinds:
                    generated.append(cache.get(query, kind))
                report("Expansões prontas", complete=True, reused=all(item["cache_hit"] for item in generated))
        output.mkdir(parents=True, exist_ok=True)
        (output / "qexp_generation_stage.json").write_text(
            json.dumps(cache.records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {"generations": len(cache.records), "cache_hits": sum(r["cache_hit"] for r in cache.records)}
    if stage != "run":
        raise ValueError(f"Etapa desconhecida: {stage}")
    from .checkpoints import run_with_checkpoints, save_result
    from .evaluation import qexp_export
    from .reranking import BGEReranker
    result = run_with_checkpoints(experiments, config, cache, BGEReranker(config),
                                  settings["checkpoint_dir"], resume=resume)
    report("Salvando snapshot", query_id="")
    save_result(result, output / "qexp_run.json")
    report("Calculando hipóteses e exportando tabelas")
    return qexp_export(result, config, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "generate", "run", "analyze"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-resume", action="store_true", help="Reexecuta rankings; mantém o cache de geração.")
    parser.add_argument("--progress", choices=["auto", "plain", "off"], default="auto")
    args = parser.parse_args()
    result = execute_stage(args.stage, load_settings(args.config), resume=not args.no_resume, progress=args.progress)
    print(result["summary"].to_string(index=False) if "summary" in result else result)


if __name__ == "__main__":
    main()
