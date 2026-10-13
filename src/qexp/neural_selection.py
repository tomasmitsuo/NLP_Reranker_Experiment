"""MLP de utilidade opcional: treino CPU, inferência NumPy e pesos JSON.

Mesmos alvos e decisão do Ridge; nenhuma expansão é executada no treinamento.
Cada semente gera uma política independente, nunca selecionada pelo teste.
"""
import copy
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .benefit_selection import macro_mean
from .config import FIXED_METHODS, ALL_TECHNIQUES
from .learned_selection import protocol
from .selector_features import feature_names
from .utility_selection import UtilitySelector, utility_table


class NeuralUtilitySelector(UtilitySelector):
    """Interface de L preservada; apenas a estimativa dos componentes muda."""

    def __init__(self, artifact):
        a = artifact
        features = feature_names(a.get("feature_set"))
        methods = a.get("methods", [])
        allowed = ALL_TECHNIQUES if a.get("selection_scope") == "all_techniques" else FIXED_METHODS
        if (a.get("schema") != 4 or a.get("estimator") != "mlp"
                or not methods or methods != sorted(set(methods))
                or a.get("selection_scope", "fixed") not in {"fixed", "all_techniques"}
                or not set(methods) <= allowed or a.get("features") != features
                or a.get("method") != (methods[0] if len(methods) == 1 else "portfolio")
                or a.get("activation") != "relu"
                or a.get("output_order") != [[m, c] for m in methods for c in ("gain", "harm", "cost")]
                or type(a.get("hidden_units")) is not int or a["hidden_units"] < 1):
            raise ValueError("Artefato neural incompatível.")
        for name, size, positive in (("mean", len(features), False), ("scale", len(features), True),
                                     ("target_mean", 3*len(methods), False), ("target_scale", 3*len(methods), True)):
            v = np.asarray(a[name], dtype=float)
            if v.shape != (size,) or not np.isfinite(v).all() or (positive and np.any(v <= 0)):
                raise ValueError("Normalização neural inválida.")
        if (not np.isfinite([a["threshold"], a["harm_penalty"], a["cost_weight"]]).all()
                or min(a["threshold"], a["harm_penalty"], a["cost_weight"]) < 0):
            raise ValueError("Utilidade neural inválida.")
        sizes = [len(features), a["hidden_units"], 3*len(methods)]
        if len(a["layers"]) != 2:
            raise ValueError("Camadas neurais inválidas.")
        for layer, inputs, outputs in zip(a["layers"], sizes, sizes[1:]):
            w, b = np.asarray(layer["weight"]), np.asarray(layer["bias"])
            if (w.shape != (outputs, inputs) or b.shape != (outputs,)
                    or not np.isfinite(w).all() or not np.isfinite(b).all()):
                raise ValueError("Pesos neurais inválidos.")
        self.artifact = copy.deepcopy(a)
        self.fit_ids = {tuple(pair) for pair in a["fit_ids"]}
        self.fit_hashes = set(a["fit_query_hashes"])

    def components(self, features):
        a = self.artifact
        x = np.asarray([features[f] for f in a["features"]], dtype=float)
        if not np.isfinite(x).all():
            raise ValueError("Atributos não finitos.")
        x = (x-a["mean"])/a["scale"]
        for i, layer in enumerate(a["layers"]):
            x = x @ np.asarray(layer["weight"]).T + layer["bias"]
            if i == 0:
                x = np.maximum(x, 0)
        values = x*a["target_scale"] + a["target_mean"]
        if not np.isfinite(values).all():
            raise ValueError("Previsões neurais não finitas.")
        return {m: {c: float(values[3*i+j]) if c == "gain" else max(0., float(values[3*i+j]))
                    for j, c in enumerate(("gain", "harm", "cost"))}
                for i, m in enumerate(a["methods"])}


def fit_neural_selector(result, splits, methods=("G", "K"), feature_set="bge",
                        harm_penalty=1., cost_weight=.001, *, seed=42, hidden_units=16,
                        weight_decays=(.001, .01), learning_rate=.01, max_epochs=300,
                        patience=30, frame=None, selection_scope="fixed"):
    """Treino completo por coleção; early stopping na validação explícita.

Normalização de entradas/alvos só no treino. Época por SmoothL1 de validação;
weight decay e limiar por utilidade de validação. Não usa partição de teste.
"""
    import torch
    from sklearn.preprocessing import StandardScaler

    methods = sorted(set(methods))
    features = feature_names(feature_set)
    allowed = ALL_TECHNIQUES if selection_scope == "all_techniques" else FIXED_METHODS
    if (selection_scope not in {"fixed", "all_techniques"} or not methods or not set(methods) <= allowed
            or not np.isfinite([harm_penalty, cost_weight, learning_rate]).all()
            or min(harm_penalty, cost_weight) < 0 or learning_rate <= 0
            or any(type(v) is not int or v < 1 for v in (hidden_units, max_epochs, patience))
            or type(seed) is not int or not 0 <= seed < 2**32
            or not weight_decays or not np.isfinite(weight_decays).all() or min(weight_decays) < 0):
        raise ValueError("Parâmetros neurais inválidos.")
    frame = utility_table(result, methods, splits, selection_scope) if frame is None else frame.copy()
    if not set(features) <= set(frame):
        raise ValueError("Snapshot sem atributos solicitados; raridade exige collect_query_statistics=true na coleta.")
    train, val = [frame[frame.split == p] for p in ("train", "validation")]
    if train.empty or val.empty:
        raise ValueError("Treino e validação não podem ser vazios.")
    columns = features + [f"{p}_{m}" for m in methods for p in ("delta", "extra_sec")]
    for part in (train, val):
        if (not np.isfinite(part[columns].to_numpy(float)).all()
                or (part[[f"extra_sec_{m}" for m in methods]] < 0).any().any()):
            raise ValueError("Atributos, ganhos ou custos inválidos em desenvolvimento.")

    def targets(part):
        return np.column_stack([v for m in methods for v in
                                (part[f"delta_{m}"], np.maximum(-part[f"delta_{m}"], 0), part[f"extra_sec_{m}"])])

    scaler = StandardScaler().fit(train[features])
    target_scaler = StandardScaler().fit(targets(train))
    tensor = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64, device="cpu")
    x, vx = [tensor(scaler.transform(p[features])) for p in (train, val)]
    y, vy = [tensor(target_scaler.transform(targets(p))) for p in (train, val)]

    def weights(part):
        counts = part.dataset.value_counts()
        return tensor(part.dataset.map(lambda d: len(part)/(len(counts)*counts[d])).to_numpy())

    w, vw = weights(train), weights(val)
    loss_fn = torch.nn.SmoothL1Loss(reduction="none", beta=1.)
    trials, history, candidates = [], [], {}
    for decay in sorted(set(weight_decays)):
        # Restaura RNG do chamador; treinamento e tensores explicitamente em CPU.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            net = torch.nn.Sequential(torch.nn.Linear(len(features), hidden_units, device="cpu", dtype=torch.float64),
                                      torch.nn.ReLU(), torch.nn.Linear(hidden_units, 3*len(methods), device="cpu", dtype=torch.float64))
            optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=decay)
            best_loss, best_state, best_epoch, stalled = float("inf"), None, 0, 0
            for epoch in range(1, max_epochs+1):
                net.train()
                optimizer.zero_grad()
                loss = (loss_fn(net(x), y).mean(dim=1)*w).mean()
                loss.backward()
                optimizer.step()
                net.eval()
                with torch.no_grad():
                    vl = float((loss_fn(net(vx), vy).mean(dim=1)*vw).mean())
                if not np.isfinite([float(loss.detach()), vl]).all():
                    raise ValueError("Treinamento neural produziu perda não finita.")
                history.append({"weight_decay": float(decay), "epoch": epoch,
                                "train_loss": float(loss.detach()), "validation_loss": vl})
                if vl < best_loss-1e-8:
                    best_loss, best_epoch, stalled = vl, epoch, 0
                    best_state = copy.deepcopy(net.state_dict())
                else:
                    stalled += 1
                if stalled >= patience:
                    break
            net.load_state_dict(best_state)
            layers = [{"weight": net[i].weight.detach().numpy().tolist(),
                       "bias": net[i].bias.detach().numpy().tolist()} for i in (0, 2)]
        used = frame[frame.split != "test"]
        artifact = {"schema": 4, "estimator": "mlp", "method": methods[0] if len(methods) == 1 else "portfolio",
                    "methods": methods, "feature_set": feature_set, "features": features,
                    "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                    "target_mean": target_scaler.mean_.tolist(), "target_scale": target_scaler.scale_.tolist(),
                    "activation": "relu", "hidden_units": hidden_units, "layers": layers,
                    "output_order": [[m, c] for m in methods for c in ("gain", "harm", "cost")],
                    "threshold": 0., "harm_penalty": float(harm_penalty), "cost_weight": float(cost_weight),
                    "seed": seed, "protocols": {m: protocol(result["metadata"]["config"], m) for m in methods},
                    "fit_ids": used[["dataset", "query_id"]].values.tolist(),
                    "fit_query_hashes": sorted(set(used.query_hash)),
                    "test_ids": frame.loc[frame.split == "test", ["dataset", "query_id"]].values.tolist(),
                    "split_sha256": hashlib.sha256(Path(splits).read_bytes()).hexdigest(),
                    "training": {"device": "cpu", "dtype": "float64", "torch_version": torch.__version__,
                                 "torch_num_threads": torch.get_num_threads(),
                                 "optimizer": "AdamW", "loss": "SmoothL1", "beta": 1., "learning_rate": learning_rate,
                                 "max_epochs": max_epochs, "patience": patience, "min_delta": 1e-8,
                                 "weight_decays": list(map(float, weight_decays)), "collection_balanced": True},
                    "cost_unit": "segundos adicionais compostos históricos em relação a B; mesma configuração de hardware",
                    "note": "Normalização só no treino; época, regularização e limiar só na validação. Semente não selecionada pelo teste."}
        if selection_scope != "fixed":
            artifact["selection_scope"] = selection_scope
        model = NeuralUtilitySelector(artifact)
        scores = np.asarray([list(model.scores(r).values()) for r in val.to_dict("records")])
        chosen = scores.argmax(axis=1)
        predicted = scores[np.arange(len(val)), chosen]
        gains = val[[f"delta_{m}" for m in methods]].to_numpy()[np.arange(len(val)), chosen]
        costs = val[[f"extra_sec_{m}" for m in methods]].to_numpy()[np.arange(len(val)), chosen]
        candidates[decay] = artifact
        for threshold in sorted({0., float(max(0, predicted.max())+1), *predicted[predicted >= 0].tolist()}):
            expand = predicted > threshold
            utility = expand*(gains-harm_penalty*np.maximum(-gains, 0)-cost_weight*costs)
            trials.append({"weight_decay": float(decay), "threshold": float(threshold), "best_epoch": best_epoch,
                           "validation_loss": best_loss, "validation_utility": macro_mean(utility, val.dataset),
                           "validation_gain": macro_mean(gains*expand, val.dataset),
                           "expansion_rate": macro_mean(expand, val.dataset),
                           "validation_extra_sec": macro_mean(costs*expand, val.dataset)})
    best = max(trials, key=lambda r: (r["validation_utility"], -r["expansion_rate"], r["weight_decay"], r["threshold"]))
    artifact = candidates[best["weight_decay"]]
    artifact.update(threshold=best["threshold"], validation_choice=best, training_history=history)
    return artifact, pd.DataFrame(trials)
