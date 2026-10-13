"""Treino supervisionado e replay de expansão seletiva, com partições explícitas.

python -m qexp.learned_selection --help
O artefato é JSON numérico: a inferência não desserializa objetos Python.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import QEXP_CONFIG, CONTROLLED_METHODS
from .selector_features import LEXICAL_FEATURES, BGE_FEATURES, RARITY_FEATURES, lexical_features, query_hash

METHODS = {"A", "B", "C", "C_H4", "D", "F", "G", "K", *CONTROLLED_METHODS}


def protocol(config, method):
    from .naming import normalize_config, internal_id
    config, method = normalize_config(config), internal_id(method)
    config = {**QEXP_CONFIG, **config}
    keys = ["candidate_n", "eval_k", "signal_top_k", "entropy_temperature",
            "reranker_model", "reranker_revision", "reranker_max_length"]
    if config.get("selector_baseline_first"):
        keys.append("selector_baseline_first")
    if method in {"C", "C_H4", "D", "K", *CONTROLLED_METHODS}:
        keys += ["model_name", "model_revision", "seed", "max_new_tokens", "generation_format"]
    if method in {"K", *CONTROLLED_METHODS} and (config["keyword_prompt_version"] != 1 or config["keyword_retries"]):
        keys += ["keyword_prompt_version", "keyword_retries"]
    keys += {**{m: ["keyword_count", "keyword_max_words", "rrf_k", "controlled_alpha"] for m in CONTROLLED_METHODS},
             "A": [], "B": [],
             "C_H4": ["query_repetitions", "query2doc_words", "query2doc_examples"],
             "C": ["query_repetitions", "query2doc_words", "query2doc_examples"],
             "D": ["num_reformulations", "rrf_k"],
             "F": ["rm3_fb_docs", "rm3_fb_terms", "rm3_original_weight", "rm3_temperature"],
             "G": [k for k in config if k.startswith("bert_qe_")],
             "K": ["keyword_count", "keyword_max_words", "rrf_k"]}[method]
    return {"feature_version": 2 if config["collect_query_statistics"] else 1, "method": method, **{k: config[k] for k in keys}}


def training_table(result, method):
    """Métricas são alvos; somente a lista explícita de atributos entra no modelo."""
    if method not in METHODS:
        raise ValueError("Método deve ser C, D, F, G ou K.")
    metric = f"ndcg@{result['metadata']['config']['eval_k']}"
    metrics = result["metrics"]
    if metrics.duplicated(["dataset", "query_id", "pipeline_id"]).any():
        raise ValueError("Resultados duplicados.")
    base = metrics[metrics.pipeline_id == "B"].set_index(["dataset", "query_id"])
    expanded = metrics[metrics.pipeline_id == method].set_index(["dataset", "query_id"])
    if base.empty or set(base.index) != set(expanded.index):
        raise ValueError("Método e B devem conter exatamente as mesmas consultas.")
    artifacts = {(a["dataset"], a["query_id"]): a for a in result["artifacts"] if a["pipeline_id"] == "B"}
    decisions = result["decisions"]
    rows = []
    for key, row in base.iterrows():
        saved = artifacts.get(key, {}).get("selector_features")
        if saved is None:
            # Permite diagnóstico lexical da execução histórica, sem inventar atributos BGE.
            found = decisions[(decisions.dataset == key[0]) & (decisions.query_id == key[1])]
            if found.empty:
                raise ValueError("Snapshot sem atributos; execute novamente com exportação de features.")
            saved = lexical_features(row.query_text, found.iloc[0].to_dict())
        other = expanded.loc[key]
        rows.append({"dataset": key[0], "query_id": str(key[1]), "query_text": row.query_text,
                     "query_hash": query_hash(row.query_text),
                     **{f: saved[f] for f in LEXICAL_FEATURES + BGE_FEATURES + RARITY_FEATURES if f in saved},
                     "baseline_ndcg": float(row[metric]), "expanded_ndcg": float(other[metric]),
                     "delta_ndcg": float(other[metric] - row[metric]),
                     "generation_status": other.generation_status})
    return pd.DataFrame(rows)


def attach_splits(frame, path):
    splits = pd.read_csv(path, dtype=str, keep_default_na=False)
    keys = ["dataset", "query_id"]
    if not set(keys + ["split"]) <= set(splits) or splits.duplicated(keys).any():
        raise ValueError("Split exige dataset, query_id, split, sem duplicatas.")
    if set(splits.split) != {"train", "validation", "test"}:
        raise ValueError("Defina partições não vazias train, validation e test.")
    if set(map(tuple, splits[keys].values)) != set(map(tuple, frame[keys].values)):
        raise ValueError("Manifesto de splits deve cobrir exatamente as consultas do snapshot.")
    columns = keys + ["split"] + (["group"] if "group" in splits else [])
    joined = frame.merge(splits[columns], on=keys, validate="one_to_one")
    for key in ["query_hash"] + (["group"] if "group" in joined else []):
        if key == "group" and (joined[key] == "").any():
            raise ValueError("Grupos de paráfrases não podem ser vazios.")
        if (joined.groupby(key).split.nunique() > 1).any():
            raise ValueError(f"Vazamento: {key} aparece em mais de uma partição.")
    return joined


def fit_selector(result, splits_path, method="D", feature_set="lexical", epsilon=0.0, *, allow_constant=False):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if feature_set not in {"lexical", "bge"} or not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("Feature set/epsilon inválido.")
    frame = attach_splits(training_table(result, method), splits_path)
    features = LEXICAL_FEATURES + (BGE_FEATURES if feature_set == "bge" else [])
    if not set(features) <= set(frame):
        raise ValueError("Snapshot histórico não tem atributos BGE; gere uma nova execução.")
    train, validation = [frame[frame.split == split] for split in ("train", "validation")]
    y = (train.delta_ndcg > epsilon).astype(int)
    if y.nunique() != 2 and not allow_constant:
        raise ValueError("Treino precisa de consultas com e sem benefício da expansão.")
    if not np.isfinite(frame[features].to_numpy(dtype=float)).all():
        raise ValueError("Atributos não finitos.")
    scaler = StandardScaler().fit(train[features])
    estimator = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    if y.nunique() == 2:
        estimator.fit(scaler.transform(train[features]), y)
    else:
        estimator.coef_ = np.zeros((1, len(features)))
        estimator.intercept_ = np.array([30. if y.iloc[0] else -30.])
    z = scaler.transform(validation[features]) @ estimator.coef_[0] + estimator.intercept_[0]
    probabilities = np.clip(1/(1+np.exp(-np.clip(z, -700, 700))),
                            np.finfo(float).eps, 1 - np.finfo(float).eps)
    thresholds = sorted(set([0.0, 1.0, *probabilities.tolist()]))
    choices = []
    for threshold in thresholds:
        selected = probabilities >= threshold
        choices.append({"threshold": threshold,
                        "mean_delta_ndcg": float(np.mean(validation.delta_ndcg.to_numpy() * selected)),
                        "expansion_rate": float(np.mean(selected))})
    # Maior ganho médio na validação; em empates, menos expansões.
    best = max(choices, key=lambda x: (x["mean_delta_ndcg"], -x["expansion_rate"], x["threshold"]))
    used = frame[frame.split != "test"]
    artifact = {
        "schema": 1, "method": method, "features": features, "feature_set": feature_set,
        "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "coef": estimator.coef_[0].tolist(), "intercept": float(estimator.intercept_[0]),
        "threshold": best["threshold"], "epsilon": epsilon,
        "protocol": protocol(result["metadata"]["config"], method),
        "fit_ids": used[["dataset", "query_id"]].values.tolist(),
        "fit_query_hashes": sorted(set(used.query_hash)),
        "test_ids": frame.loc[frame.split == "test", ["dataset", "query_id"]].values.tolist(),
        "split_sha256": hashlib.sha256(Path(splits_path).read_bytes()).hexdigest(),
        "training_queries": len(train), "validation_queries": len(validation),
        "validation_choice": best,
        "note": "Limiar escolhido somente na validação. Dados já explorados não se tornam teste inédito.",
    }
    return artifact, pd.DataFrame(choices), used


class LearnedSelector:
    def __init__(self, artifact):
        self.artifact = artifact
        expected = LEXICAL_FEATURES + (BGE_FEATURES if artifact.get("feature_set") == "bge" else [])
        if (artifact.get("schema") != 1 or artifact.get("method") not in METHODS
                or artifact.get("feature_set") not in {"lexical", "bge"}
                or artifact.get("features") != expected):
            raise ValueError("Artefato de seletor incompatível.")
        for field in ("mean", "scale", "coef"):
            values = np.asarray(artifact[field], dtype=float)
            if values.shape != (len(expected),) or not np.isfinite(values).all():
                raise ValueError(f"Parâmetros inválidos: {field}")
        if (np.any(np.asarray(artifact["scale"]) <= 0)
                or not np.isfinite(artifact["intercept"])
                or not 0 <= artifact["threshold"] <= 1):
            raise ValueError("Escala, intercepto ou limiar inválido.")
        self.fit_ids = {tuple(pair) for pair in artifact["fit_ids"]}
        self.fit_hashes = set(artifact["fit_query_hashes"])

    @classmethod
    def load(cls, path):
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        from .naming import translate_model
        artifact = translate_model(artifact)
        if artifact.get("schema") == 4:
            from .neural_selection import NeuralUtilitySelector
            return NeuralUtilitySelector(artifact)
        if artifact.get("schema") == 3:
            from .utility_selection import UtilitySelector
            return UtilitySelector(artifact)
        if artifact.get("schema") == 2:
            from .benefit_selection import BenefitSelector
            return BenefitSelector(artifact)
        return cls(artifact)

    def choose(self, features):
        expand, probability = self.predict(features)
        return self.artifact["method"] if expand else "B", probability

    def validate_protocol(self, config):
        if (config["selective_method"] != self.artifact["method"]
                or protocol(config, self.artifact["method"]) != self.artifact["protocol"]):
            raise ValueError("Protocolo do seletor difere da execução; treine para esta configuração.")

    def check_unseen(self, dataset, query_id, query):
        if (dataset, str(query_id)) in self.fit_ids or query_hash(query) in self.fit_hashes:
            raise ValueError("Vazamento: consulta de treino/validação na avaliação do seletor.")

    def predict(self, features):
        model = self.artifact
        values = np.asarray([features[name] for name in model["features"]], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Atributos não finitos.")
        z = float(np.dot((values - model["mean"]) / model["scale"], model["coef"]) + model["intercept"])
        probability = float(np.clip(1 / (1 + np.exp(-np.clip(z, -700, 700))),
                                    np.finfo(float).eps, 1 - np.finfo(float).eps))
        return probability >= model["threshold"], probability


def evaluate_selector(result, model, splits_path):
    model.validate_protocol({**QEXP_CONFIG, **result["metadata"]["config"],
                             "selective_method": model.artifact["method"]})
    if hashlib.sha256(Path(splits_path).read_bytes()).hexdigest() != model.artifact["split_sha256"]:
        raise ValueError("Use o mesmo manifesto de splits do treinamento.")
    if model.artifact["schema"] in {2, 3, 4}:
        from .benefit_selection import policy_table
        from .selector_study import replay_policy, summarize_policies
        frame = policy_table(result, model.artifact["methods"], splits_path)
        if "group" not in frame:
            frame["group"] = frame.query_hash
        predictions = replay_policy(result, model, frame, model.artifact.get("policy_name", "L"))
        summary, _, _ = summarize_policies(predictions, seed=model.artifact["seed"])
        return predictions, summary
    frame = attach_splits(training_table(result, model.artifact["method"]), splits_path)
    test = frame[frame.split == "test"].copy()
    probabilities, decisions = [], []
    for row in test.to_dict("records"):
        model.check_unseen(row["dataset"], row["query_id"], row["query_text"])
        decision, probability = model.predict(row)
        probabilities.append(probability)
        decisions.append(decision)
    test["probability"] = probabilities
    test["expand"] = decisions
    test["selected_ndcg"] = test.baseline_ndcg + test.delta_ndcg * test.expand
    summary = []
    for dataset, group in test.groupby("dataset"):
        rng = np.random.default_rng(42)
        gain = (group.delta_ndcg * group.expand).to_numpy()
        bootstrap = [rng.choice(gain, size=len(gain), replace=True).mean() for _ in range(1000)]
        # Mesma quantidade exata de solicitações por dataset, 100 sorteios independentes.
        random_gains = []
        for _ in range(100):
            selected = rng.choice(len(group), size=int(group.expand.sum()), replace=False)
            random_gains.append(float(group.delta_ndcg.iloc[selected].sum() / len(group)))
        summary.append({"dataset": dataset, "queries": len(group),
                        "baseline_ndcg": group.baseline_ndcg.mean(),
                        "always_ndcg": group.expanded_ndcg.mean(),
                        "selected_ndcg": group.selected_ndcg.mean(),
                        "oracle_ndcg": np.maximum(group.baseline_ndcg, group.expanded_ndcg).mean(),
                        "random_matched_ndcg": group.baseline_ndcg.mean() + np.mean(random_gains),
                        "expansion_rate": group.expand.mean(),
                        "degradation_rate": float(np.mean(gain < -1e-12)),
                        "mean_delta": gain.mean(), "ci95_low": np.quantile(bootstrap, .025),
                        "ci95_high": np.quantile(bootstrap, .975)})
    return test, pd.DataFrame(summary)


def make_splits(result, seed=42):
    base = result["metrics"].query("pipeline_id == 'B'")
    frame = base[["dataset", "query_id", "query_text"]].copy()
    frame["group"] = frame.query_text.map(query_hash)
    groups = np.array(sorted(frame.group.unique()))
    if len(groups) < 5:
        raise ValueError("São necessários ao menos cinco grupos de consultas para três partições.")
    np.random.default_rng(seed).shuffle(groups)
    n_train, n_val = max(1, int(.6 * len(groups))), max(1, int(.2 * len(groups)))
    mapping = {g: ("train" if i < n_train else "validation" if i < n_train + n_val else "test")
               for i, g in enumerate(groups)}
    frame["split"] = frame.group.map(mapping)
    return frame[["dataset", "query_id", "split", "group"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["split", "train", "evaluate"])
    parser.add_argument("--run", required=True, type=Path, help="Snapshot qexp_run.json")
    parser.add_argument("--output", required=True, type=Path, help="Nova pasta; não sobrescreve experimentos")
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--method", choices=sorted(METHODS), default="D")
    parser.add_argument("--features", choices=["lexical", "bge"], default="lexical")
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    if args.stage != "split" and args.splits is None:
        parser.error("train/evaluate exigem --splits")
    if args.stage == "evaluate" and args.model is None:
        parser.error("evaluate exige --model")
    from .checkpoints import load_result
    result = load_result(args.run)
    if args.output.exists():
        raise ValueError("Use uma pasta de saída nova.")
    if args.stage == "split":
        frame = make_splits(result, args.seed)
        args.output.mkdir(parents=True)
        frame.to_csv(args.output / "splits.csv", index=False)
    elif args.stage == "train":
        model, thresholds, used = fit_selector(result, args.splits, args.method, args.features, args.epsilon)
        model["source_sha256"] = hashlib.sha256(args.run.read_bytes()).hexdigest()
        args.output.mkdir(parents=True)
        (args.output / "selector.json").write_text(json.dumps(model, indent=2, allow_nan=False))
        thresholds.to_csv(args.output / "validation_thresholds.csv", index=False)
        used.to_csv(args.output / "training_validation.csv", index=False)
    else:
        predictions, summary = evaluate_selector(result, LearnedSelector.load(args.model), args.splits)
        args.output.mkdir(parents=True)
        predictions.to_csv(args.output / "test_predictions.csv", index=False)
        summary.to_csv(args.output / "test_summary.csv", index=False)
        print(summary.to_string(index=False))
    print(args.output)


if __name__ == "__main__":
    main()
