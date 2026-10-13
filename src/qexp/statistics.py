"""Inferência pareada por grupos de consultas; nenhuma escolha de modelo aqui."""
import hashlib
import itertools

import numpy as np
import pandas as pd


def stable_seed(seed, name):
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:4], "little")


def holm(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("p-valores inválidos.")
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    return adjusted


def paired_statistics(differences, groups=None, *, seed=42, resamples=2000, permutations=9999):
    """Média por consulta; reamostra ou troca sinais de grupos inteiros.

    IC percentil marginal. Teste bilateral de troca de sinais sob hipótese de
    permutabilidade pareada no grupo; não pressupõe que queries sejam i.i.d.
    quando o manifesto identifica dependências. Não prova equivalência.
    """
    d = np.asarray(differences, dtype=float)
    if d.ndim != 1 or not len(d) or not np.isfinite(d).all() or resamples < 1 or permutations < 1:
        raise ValueError("Diferenças finitas e números positivos de reamostragens são obrigatórios.")
    d = np.where(np.abs(d) <= 1e-12, 0, d)
    groups = np.arange(len(d)) if groups is None else np.asarray(groups)
    if len(groups) != len(d) or pd.isna(groups).any():
        raise ValueError("Grupos incompatíveis com os pares.")
    aggregate = pd.DataFrame({"d": d, "group": groups}).groupby("group").d.agg(["sum", "size"])
    sums, sizes = aggregate["sum"].to_numpy(), aggregate["size"].to_numpy()
    rng = np.random.default_rng(seed)
    samples = rng.integers(len(sums), size=(resamples, len(sums)))
    bootstrap = sums[samples].sum(axis=1) / sizes[samples].sum(axis=1)
    # Grupos com soma zero não alteram o teste da diferença média.
    nonzero = sums[np.abs(sums) > 1e-12]
    observed = abs(d.sum())
    if len(nonzero) <= 15:
        null = np.array([abs(np.dot(signs, nonzero)) for signs in itertools.product((-1, 1), repeat=len(nonzero))])
        p_value = float(np.mean(null >= observed - 1e-12))
        exact = True
    else:
        signs = rng.choice([-1, 1], size=(permutations, len(nonzero)))
        null = np.abs(signs @ nonzero)
        p_value = float((1 + (null >= observed - 1e-12).sum()) / (1 + permutations))
        exact = False
    return {"queries": len(d), "groups": len(sums), "mean_delta": float(d.mean()),
            "ci95_low": float(np.quantile(bootstrap, .025)), "ci95_high": float(np.quantile(bootstrap, .975)),
            "gain_rate": float(np.mean(d > 0)), "tie_rate": float(np.mean(d == 0)),
            "harm_rate": float(np.mean(d < 0)), "mean_positive_gain": float(np.maximum(d, 0).mean()),
            "mean_harm": float(np.maximum(-d, 0).mean()), "worst_delta": float(d.min()),
            "p_value": p_value, "permutation_exact": exact}
