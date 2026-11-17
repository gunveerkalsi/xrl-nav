"""Quantitative checks that explanations reflect what the policy actually uses.

* ``deletion_curve``      : replace the k most-attributed features by the baseline and track the
                            target-action probability and the action-flip rate; compare to random order.
                            A faithful method degrades the decision much faster than random.
* ``randomization_check`` : model-parameter randomisation sanity check (Adebayo et al., 2018).
                            Attributions from a randomly re-initialised network should *not* resemble
                            those of the trained one (low rank correlation = pass).
* ``method_agreement``    : Spearman rank correlation of |attribution| between methods, incl. attention.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from .saliency import _target_value, attribute


@torch.no_grad()
def _prob(model, X, T):
    return _target_value(model, torch.as_tensor(X), torch.as_tensor(T), prob=True).numpy()


def deletion_curve(model, X, actions, attr: np.ndarray | None, baseline, k_max: int | None = None,
                   seed: int = 0) -> pd.DataFrame:
    """attr=None gives the random-order control."""
    X = np.asarray(X, dtype=np.float32)
    N, D = X.shape
    k_max = k_max or D
    rng = np.random.default_rng(seed)
    if attr is None:
        order = np.stack([rng.permutation(D) for _ in range(N)])
    else:
        order = np.argsort(-np.abs(attr), 1)
    T = np.asarray(actions, dtype=np.int64)
    p0 = _prob(model, X, T)
    a0 = model.probs_np(X).argmax(1) if not model.continuous else None
    rows = []
    Xk = X.copy()
    b = np.broadcast_to(baseline, X.shape)
    for k in range(0, k_max + 1):
        if k > 0:
            idx = order[:, k - 1]
            Xk[np.arange(N), idx] = b[np.arange(N), idx]
        pk = _prob(model, Xk, T)
        row = {"k": k, "prob": float(pk.mean()), "prob_drop": float((p0 - pk).mean())}
        if a0 is not None:
            row["flip_rate"] = float((model.probs_np(Xk).argmax(1) != a0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def deletion_auc(curve: pd.DataFrame, k_max: int = 10) -> float:
    """Mean probability drop over the first k_max deletions (higher = more faithful)."""
    c = curve[(curve.k >= 1) & (curve.k <= k_max)]
    return float(c.prob_drop.mean())


def _randomized_copy(model):
    m = copy.deepcopy(model)
    gen = torch.Generator().manual_seed(123)
    with torch.no_grad():
        for p_ in m.policy.parameters():
            if p_.dim() > 1:
                p_.normal_(0, 1.0 / np.sqrt(p_.shape[1]), generator=gen)
            else:
                p_.normal_(0, 0.1, generator=gen)
    return m


def randomization_check(model, X, actions, baseline, methods=("gradient", "integrated_grads", "occlusion"),
                        groups: dict | None = None):
    rnd = _randomized_copy(model)
    rows = []
    if groups is not None:
        from .saliency import group_shapley

        a = np.abs(group_shapley(model, X, actions, groups, baseline))
        r = np.abs(group_shapley(rnd, X, actions, groups, baseline))
        rho = [spearmanr(a[i], r[i]).statistic for i in range(len(X))]
        rows.append({"method": "group_shapley", "spearman_trained_vs_random": float(np.nanmean(rho)),
                     "passes": bool(np.nanmean(rho) < 0.5)})
    for m in methods:
        a = np.abs(attribute(model, X, actions, method=m, baseline=baseline))
        r = np.abs(attribute(rnd, X, actions, method=m, baseline=baseline))
        rho = [spearmanr(a[i], r[i]).statistic for i in range(len(X))]
        rows.append({"method": m, "spearman_trained_vs_random": float(np.nanmean(rho)),
                     "passes": bool(np.nanmean(rho) < 0.5)})
    return pd.DataFrame(rows)


def method_agreement(attrs: dict[str, np.ndarray]) -> pd.DataFrame:
    names = list(attrs)
    M = np.eye(len(names))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j <= i:
                continue
            A, B = np.abs(attrs[a]), np.abs(attrs[b])
            rho = np.nanmean([spearmanr(A[k], B[k]).statistic for k in range(len(A))])
            M[i, j] = M[j, i] = rho
    return pd.DataFrame(M, index=names, columns=names)
