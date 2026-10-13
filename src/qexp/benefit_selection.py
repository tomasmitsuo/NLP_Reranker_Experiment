"""Seleção de expansão orientada ao ganho final do BGE, antes de gerar texto.

Ridge estima delta de nDCG (ou utilidade penalizada); logística ponderada dá
maior peso a erros com maior impacto. Modelos portáveis, sem pickle.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .learned_selection import METHODS, attach_splits, protocol, training_table
from .selector_features import LEXICAL_FEATURES, BGE_FEATURES, query_hash, FEATURE_SETS, feature_names


def policy_table(result, methods, splits_path):
    methods = sorted(set(methods))
    if not methods or not set(methods) <= METHODS:
        raise ValueError("Métodos devem pertencer a C/D/F/G/K.")
    frame = attach_splits(training_table(result, methods[0]), splits_path)
    for method in methods:
        target = training_table(result, method)[["dataset", "query_id", "expanded_ndcg", "delta_ndcg"]]
        frame = frame.merge(target.rename(columns={"expanded_ndcg": f"ndcg_{method}",
                                                   "delta_ndcg": f"delta_{method}"}),
                            on=["dataset", "query_id"], validate="one_to_one")
    return frame


def macro_mean(values, datasets):
    """Cada coleção pesa igualmente no ajuste, apesar dos diferentes tamanhos."""
    return float(pd.Series(np.asarray(values)).groupby(np.asarray(datasets)).mean().mean())


def fit_benefit_selector(result, splits_path, methods=("G", "K"), feature_set="bge",
                         estimator="ridge", harm_penalty=0.0, max_expansion_rate=1.0,
                         regularizations=(0.1, 1.0, 10.0, 100.0), seed=42, frame=None):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler

    methods = sorted(set(methods))
    if (feature_set not in FEATURE_SETS or estimator not in {"ridge", "weighted_logistic"}
            or not np.isfinite(harm_penalty) or harm_penalty < 0
            or not 0 <= max_expansion_rate <= 1
            or not regularizations or any(not np.isfinite(x) or x <= 0 for x in regularizations)):
        raise ValueError("Parâmetros de seleção inválidos.")
    if estimator == "weighted_logistic" and len(methods) != 1:
        raise ValueError("Logística ponderada é binária; use ridge para comparar ganhos entre métodos.")
    if not methods or not set(methods) <= METHODS:
        raise ValueError("Métodos devem pertencer a C/D/F/G/K.")
    frame = policy_table(result, methods, splits_path) if frame is None else frame.copy()
    features = feature_names(feature_set)
    train, validation = [frame[frame.split == part] for part in ("train", "validation")]
    if train.empty or validation.empty:
        raise ValueError("Treino e validação devem ser não vazios.")
    # Dados de teste não participam nem da normalização nem da seleção.
    used = frame[frame.split != "test"]
    if not np.isfinite(used[features].to_numpy(dtype=float)).all():
        raise ValueError("Atributos não finitos em desenvolvimento.")
    deltas = train[[f"delta_{m}" for m in methods]].to_numpy(dtype=float)
    val_delta = validation[[f"delta_{m}" for m in methods]].to_numpy(dtype=float)
    if not np.isfinite(deltas).all() or not np.isfinite(val_delta).all():
        raise ValueError("Alvos não finitos em desenvolvimento.")
    scaler = StandardScaler().fit(train[features])
    x, vx = scaler.transform(train[features]), scaler.transform(validation[features])
    # Balanceamento por coleção, calculado somente no treino.
    counts = train.dataset.value_counts()
    weights = train.dataset.map(lambda d: len(train) / (len(counts) * counts[d])).to_numpy()
    candidates, choices = [], []
    for regularization in regularizations:
        models = []
        for i, method in enumerate(methods):
            target = deltas[:, i] - harm_penalty * np.maximum(-deltas[:, i], 0)
            if estimator == "ridge":
                fitted = Ridge(alpha=regularization).fit(x, target, sample_weight=weights)
                coef, intercept = fitted.coef_, fitted.intercept_
            else:
                y = (deltas[:, i] > 1e-12).astype(int)
                if len(np.unique(y)) < 2:
                    # Mantém um modelo válido mesmo em coleções sem exemplos positivos.
                    coef, intercept = np.zeros(len(features)), (30.0 if y[0] else -30.0)
                else:
                    impact = np.abs(target)
                    impact = (impact + 1e-4) / (impact.mean() + 1e-4)
                    fitted = LogisticRegression(C=1 / regularization, max_iter=2000, random_state=seed)
                    fitted.fit(x, y, sample_weight=weights * impact)
                    coef, intercept = fitted.coef_[0], fitted.intercept_[0]
            models.append({"method": method, "coef": np.asarray(coef).tolist(), "intercept": float(intercept)})
        scores = np.column_stack([vx @ m["coef"] + m["intercept"] for m in models])
        if estimator == "weighted_logistic":
            scores = 1 / (1 + np.exp(-np.clip(scores, -700, 700)))
        selected = scores.argmax(axis=1)  # Empates estáveis na ordem alfabética.
        best_score = scores[np.arange(len(validation)), selected]
        realized = val_delta[np.arange(len(validation)), selected]
        thresholds = sorted(set([0.0, float(max(0, best_score.max()) + 1),
                                 *best_score[best_score >= 0].tolist()]))
        for threshold in thresholds:
            expand = best_score > threshold  # Zero previsto conserva B.
            rates = pd.Series(expand).groupby(validation.dataset.to_numpy()).mean()
            if rates.max() > max_expansion_rate + 1e-12:
                continue
            gain = realized * expand
            choice = {"regularization": float(regularization), "threshold": threshold,
                      "validation_gain": macro_mean(gain, validation.dataset),
                      "validation_utility": macro_mean(gain - harm_penalty * np.maximum(-gain, 0), validation.dataset),
                      "expansion_rate": macro_mean(expand, validation.dataset),
                      "max_dataset_expansion_rate": float(rates.max())}
            choices.append(choice)
            candidates.append(models)
    index = max(range(len(choices)), key=lambda i: (choices[i]["validation_utility"],
                -choices[i]["expansion_rate"], choices[i]["regularization"], choices[i]["threshold"]))
    best = choices[index]
    artifact = {
        "schema": 2, "method": methods[0] if len(methods) == 1 else "portfolio", "methods": methods,
        "feature_set": feature_set, "features": features, "estimator": estimator,
        "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "models": candidates[index],
        "threshold": best["threshold"], "harm_penalty": harm_penalty,
        "max_expansion_rate": max_expansion_rate, "seed": seed,
        "protocols": {m: protocol(result["metadata"]["config"], m) for m in methods},
        "fit_ids": used[["dataset", "query_id"]].values.tolist(),
        "fit_query_hashes": sorted(set(used.query_hash)),
        "test_ids": frame.loc[frame.split == "test", ["dataset", "query_id"]].values.tolist(),
        "split_sha256": hashlib.sha256(Path(splits_path).read_bytes()).hexdigest(),
        "validation_choice": best, "training_queries": len(train), "validation_queries": len(validation),
        "note": "Ganho final do BGE; ajuste macro por coleção somente em desenvolvimento. Replay exploratório.",
    }
    return artifact, pd.DataFrame(choices)


class BenefitSelector:
    def __init__(self, artifact):
        self.artifact = artifact
        methods = artifact.get("methods", [])
        expected = feature_names(artifact.get("feature_set"))
        if (artifact.get("schema") != 2 or not methods or methods != sorted(set(methods))
                or not set(methods) <= METHODS or artifact.get("features") != expected
                or artifact.get("feature_set") not in FEATURE_SETS
                or artifact.get("estimator") not in {"ridge", "weighted_logistic"}
                or artifact.get("method") != (methods[0] if len(methods) == 1 else "portfolio")
                or (artifact["estimator"] == "weighted_logistic" and len(methods) != 1)):
            raise ValueError("Artefato de benefício incompatível.")
        vectors = [artifact["mean"], artifact["scale"], *[m["coef"] for m in artifact["models"]]]
        if (any(np.shape(v) != (len(expected),) or not np.isfinite(v).all() for v in vectors)
                or np.any(np.asarray(artifact["scale"]) <= 0)
                or [m["method"] for m in artifact["models"]] != methods
                or not np.isfinite([m["intercept"] for m in artifact["models"]]).all()
                or not np.isfinite(artifact["threshold"]) or artifact["threshold"] < 0):
            raise ValueError("Coeficientes ou limiar inválidos.")
        self.fit_ids = {tuple(pair) for pair in artifact["fit_ids"]}
        self.fit_hashes = set(artifact["fit_query_hashes"])

    def validate_protocol(self, config):
        if (config["selective_method"] != self.artifact["method"]
                or any(protocol(config, m) != self.artifact["protocols"].get(m) for m in self.artifact["methods"])):
            raise ValueError("Protocolo do seletor difere da execução; treine para esta configuração.")

    def check_unseen(self, dataset, query_id, query):
        if (dataset, str(query_id)) in self.fit_ids or query_hash(query) in self.fit_hashes:
            raise ValueError("Vazamento: consulta de treino/validação na avaliação do seletor.")

    def scores(self, features):
        a = self.artifact
        values = np.asarray([features[f] for f in a["features"]], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Atributos não finitos.")
        x = (values - a["mean"]) / a["scale"]
        scores = np.array([x @ m["coef"] + m["intercept"] for m in a["models"]])
        if a["estimator"] == "weighted_logistic":
            scores = 1 / (1 + np.exp(-np.clip(scores, -700, 700)))
        return dict(zip(a["methods"], map(float, scores)))

    def choose(self, features):
        scores = self.scores(features)
        method = max(scores, key=scores.get)
        return (method if scores[method] > self.artifact["threshold"] else "B"), scores[method]

    def predict(self, features):
        method, score = self.choose(features)
        return method != "B", score
