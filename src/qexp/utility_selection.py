"""Seleção de candidatos fixos por ganho, perda esperada e segundos adicionais.

Alvos e custos observados só entram em treino/validação. A inferência recebe
exclusivamente a lista de atributos anteriores à expansão, sem qrels.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .benefit_selection import BenefitSelector, policy_table, macro_mean
from .config import FIXED_METHODS, ALL_TECHNIQUES
from .learned_selection import protocol
from .selector_features import FEATURE_SETS, feature_names


def utility_table(result, methods, splits, selection_scope="fixed"):
    methods = sorted(set(methods))
    allowed = ALL_TECHNIQUES if selection_scope == "all_techniques" else FIXED_METHODS
    if selection_scope not in {"fixed", "all_techniques"} or not methods or not set(methods) <= allowed:
        raise ValueError("Utilidade exige alternativas com candidatos fixos.")
    frame = policy_table(result, methods, splits)
    metrics = result["metrics"].set_index(["dataset", "query_id", "pipeline_id"])
    for method in sorted({"B", *methods}):
        values = [float(metrics.loc[(r.dataset, r.query_id, method), "estimated_uncached_sec"])
                  for r in frame.itertuples()]
        if not np.isfinite(values).all() or min(values) < 0:
            raise ValueError("Custos históricos devem ser finitos e não negativos.")
        frame[f"seconds_{method}"] = values
    for method in methods:
        frame[f"extra_sec_{method}"] = np.maximum(frame[f"seconds_{method}"] - frame.seconds_B, 0)
    if "group" not in frame:
        frame["group"] = frame.query_hash
    return frame


def fit_utility_selector(result, splits, methods=("G", "K"), feature_set="bge",
                         harm_penalty=1.0, cost_weight=.001,
                         regularizations=(.1, 1., 10., 100.), frame=None, selection_scope="fixed"):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    methods = sorted(set(methods))
    allowed = ALL_TECHNIQUES if selection_scope == "all_techniques" else FIXED_METHODS
    if (selection_scope not in {"fixed", "all_techniques"} or not methods or not set(methods) <= allowed or feature_set not in FEATURE_SETS
            or not np.isfinite([harm_penalty, cost_weight]).all() or min(harm_penalty, cost_weight) < 0
            or not regularizations or not np.isfinite(regularizations).all() or min(regularizations) <= 0):
        raise ValueError("Parâmetros de utilidade inválidos.")
    frame = utility_table(result, methods, splits, selection_scope) if frame is None else frame.copy()
    features = feature_names(feature_set)
    if not set(features) <= set(frame):
        raise ValueError("Snapshot sem atributos solicitados; raridade exige collect_query_statistics=true na coleta.")
    train, val = [frame[frame.split == part] for part in ("train", "validation")]
    if train.empty or val.empty:
        raise ValueError("Treino e validação não podem ser vazios.")
    for subset in (train, val):
        columns = features + [f"{prefix}_{m}" for m in methods for prefix in ("delta", "extra_sec")]
        if not np.isfinite(subset[columns].to_numpy(float)).all():
            raise ValueError("Atributos, ganhos ou custos não finitos em desenvolvimento.")
    scaler = StandardScaler().fit(train[features])
    x, vx = scaler.transform(train[features]), scaler.transform(val[features])
    counts = train.dataset.value_counts()
    weights = train.dataset.map(lambda d: len(train)/(len(counts)*counts[d])).to_numpy()
    deltas = val[[f"delta_{m}" for m in methods]].to_numpy()
    costs = val[[f"extra_sec_{m}" for m in methods]].to_numpy()
    trials, models = [], []
    for alpha in regularizations:
        components = []
        for method in methods:
            gain = train[f"delta_{method}"].to_numpy()
            targets = {"gain": gain, "harm": np.maximum(-gain, 0), "cost": train[f"extra_sec_{method}"].to_numpy()}
            entry = {"method": method}
            for name, target in targets.items():
                fit = Ridge(alpha=alpha).fit(x, target, sample_weight=weights)
                entry[name] = {"coef": fit.coef_.tolist(), "intercept": float(fit.intercept_)}
            components.append(entry)
        scores = np.column_stack([
            vx @ c["gain"]["coef"] + c["gain"]["intercept"]
            - harm_penalty*np.maximum(vx @ c["harm"]["coef"] + c["harm"]["intercept"], 0)
            - cost_weight*np.maximum(vx @ c["cost"]["coef"] + c["cost"]["intercept"], 0)
            for c in components])
        chosen = scores.argmax(axis=1)
        predicted = scores[np.arange(len(val)), chosen]
        gains, extra = deltas[np.arange(len(val)), chosen], costs[np.arange(len(val)), chosen]
        for threshold in sorted({0., float(max(0, predicted.max()) + 1), *predicted[predicted >= 0].tolist()}):
            expand = predicted > threshold
            realized = expand*(gains - harm_penalty*np.maximum(-gains, 0) - cost_weight*extra)
            trials.append({"regularization": float(alpha), "threshold": float(threshold),
                           "validation_utility": macro_mean(realized, val.dataset),
                           "validation_gain": macro_mean(gains*expand, val.dataset),
                           "expansion_rate": macro_mean(expand, val.dataset),
                           "validation_extra_sec": macro_mean(extra*expand, val.dataset)})
            models.append(components)
    best_idx = max(range(len(trials)), key=lambda i: (trials[i]["validation_utility"],
                   -trials[i]["expansion_rate"], trials[i]["regularization"], trials[i]["threshold"]))
    best, components = trials[best_idx], models[best_idx]
    used = frame[frame.split != "test"]
    artifact = {"schema": 3, "estimator": "ridge", "method": methods[0] if len(methods) == 1 else "portfolio",
                "methods": methods, "feature_set": feature_set, "features": features,
                "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                "models": [{"method": c["method"], **c["gain"]} for c in components],
                "component_models": components, "threshold": best["threshold"],
                "harm_penalty": float(harm_penalty), "cost_weight": float(cost_weight),
                "seed": result["metadata"]["config"]["seed"], "validation_choice": best,
                "protocols": {m: protocol(result["metadata"]["config"], m) for m in methods},
                "fit_ids": used[["dataset", "query_id"]].values.tolist(),
                "fit_query_hashes": sorted(set(used.query_hash)),
                "test_ids": frame.loc[frame.split == "test", ["dataset", "query_id"]].values.tolist(),
                "split_sha256": hashlib.sha256(Path(splits).read_bytes()).hexdigest(),
                "cost_unit": "segundos adicionais compostos históricos em relação a B; mesma configuração de hardware",
                "note": "Pesos de utilidade predefinidos; regularização e limiar escolhidos somente na validação. Não garante orçamento ou ausência de perdas."}
    if selection_scope != "fixed":
        artifact["selection_scope"] = selection_scope
    return artifact, pd.DataFrame(trials)


class UtilitySelector(BenefitSelector):
    def __init__(self, artifact):
        allowed = ALL_TECHNIQUES if artifact.get("selection_scope") == "all_techniques" else FIXED_METHODS
        if (artifact.get("schema") != 3 or artifact.get("estimator") != "ridge"
                or artifact.get("selection_scope", "fixed") not in {"fixed", "all_techniques"}
                or not set(artifact.get("methods", [])) <= allowed):
            raise ValueError("Modelo de utilidade incompatível.")
        super().__init__({**artifact, "schema": 2})
        if (not np.isfinite([artifact["harm_penalty"], artifact["cost_weight"]]).all()
                or min(artifact["harm_penalty"], artifact["cost_weight"]) < 0
                or [c["method"] for c in artifact["component_models"]] != artifact["methods"]):
            raise ValueError("Componentes de utilidade inválidos.")
        for c in artifact["component_models"]:
            for name in ("gain", "harm", "cost"):
                if (np.shape(c[name]["coef"]) != (len(artifact["features"]),)
                        or not np.isfinite([*c[name]["coef"], c[name]["intercept"]]).all()):
                    raise ValueError("Coeficientes de utilidade inválidos.")
        self.artifact = artifact

    def components(self, features):
        a = self.artifact
        values = np.asarray([features[f] for f in a["features"]], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Atributos não finitos.")
        x = (values-a["mean"])/a["scale"]
        return {c["method"]: {name: float(x @ c[name]["coef"] + c[name]["intercept"])
                            if name == "gain" else max(0., float(x @ c[name]["coef"] + c[name]["intercept"]))
                            for name in ("gain", "harm", "cost")} for c in a["component_models"]}

    def scores(self, features):
        a = self.artifact
        return {m: c["gain"]-a["harm_penalty"]*c["harm"]-a["cost_weight"]*c["cost"]
                for m, c in self.components(features).items()}
