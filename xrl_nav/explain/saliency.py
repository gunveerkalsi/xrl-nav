"""Post-hoc feature attribution for navigation policies.

All methods share one signature::

    attr = method(model, X, targets, **kw)   # X: (N, D) float32, targets: (N,) int -> attr: (N, D)

The explained quantity ``f`` is
    * discrete actions : log pi(a | s) for the target action a (gradient methods) and
                         pi(a | s) for perturbation methods (occlusion, Shapley)
    * continuous       : the Gaussian mean of action dimension ``target`` (0 = v, 1 = w)

Methods
    gradient          : vanilla saliency, d f / d x
    grad_x_input      : gradient * (x - baseline)
    integrated_grads  : Integrated Gradients (Sundararajan et al., 2017), completeness holds
    smoothgrad        : gradient averaged over Gaussian-perturbed inputs (Smilkov et al., 2017)
    occlusion         : f(x) - f(x with feature i set to baseline)
    group_shapley     : exact Shapley values over semantic feature groups (lidar sectors, goal, motion)
"""
from __future__ import annotations

import itertools
from math import factorial

import numpy as np
import torch


def _target_value(model, x: torch.Tensor, targets: torch.Tensor, prob: bool = False) -> torch.Tensor:
    s = model(x)
    if model.continuous:
        return s.gather(1, targets.view(-1, 1)).squeeze(1)
    t = model.temperature if model.algo == "dqn" else 1.0
    logp = torch.log_softmax(s / t, -1).gather(1, targets.view(-1, 1)).squeeze(1)
    return logp.exp() if prob else logp


def _prep(X, targets, baseline):
    X = torch.as_tensor(np.asarray(X, dtype=np.float32))
    T = torch.as_tensor(np.asarray(targets, dtype=np.int64)).view(-1)
    B = torch.as_tensor(np.asarray(baseline, dtype=np.float32)).expand_as(X) if baseline is not None else torch.zeros_like(X)
    return X, T, B


def gradient(model, X, targets, baseline=None, **_):
    X, T, _ = _prep(X, targets, baseline)
    X = X.clone().requires_grad_(True)
    _target_value(model, X, T).sum().backward()
    return X.grad.detach().numpy()


def grad_x_input(model, X, targets, baseline=None, **_):
    g = gradient(model, X, targets)
    _, _, B = _prep(X, targets, baseline)
    return g * (np.asarray(X, dtype=np.float32) - B.numpy())


def integrated_grads(model, X, targets, baseline=None, steps: int = 64, **_):
    X, T, B = _prep(X, targets, baseline)
    alphas = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps  # midpoint Riemann sum
    total = torch.zeros_like(X)
    for a in alphas:
        xi = (B + a * (X - B)).clone().requires_grad_(True)
        _target_value(model, xi, T).sum().backward()
        total += xi.grad.detach()
    return ((X - B) * total / steps).numpy()


def smoothgrad(model, X, targets, baseline=None, n_samples: int = 32, sigma: float = 0.05, seed: int = 0, **_):
    X, T, _ = _prep(X, targets, baseline)
    g = torch.Generator().manual_seed(seed)
    total = torch.zeros_like(X)
    for _ in range(n_samples):
        xi = (X + sigma * torch.randn(X.shape, generator=g)).clone().requires_grad_(True)
        _target_value(model, xi, T).sum().backward()
        total += xi.grad.detach()
    return (total / n_samples).numpy()


@torch.no_grad()
def occlusion(model, X, targets, baseline=None, **_):
    X, T, B = _prep(X, targets, baseline)
    N, D = X.shape
    f0 = _target_value(model, X, T, prob=True)
    out = torch.zeros(N, D)
    for i in range(D):
        xi = X.clone()
        xi[:, i] = B[:, i]
        out[:, i] = f0 - _target_value(model, xi, T, prob=True)
    return out.numpy()


@torch.no_grad()
def group_shapley(model, X, targets, groups: dict[str, list[int]], baseline=None, batch: int = 64, **_):
    """Exact Shapley values of the target-action probability over feature groups.

    With G groups this evaluates 2^G coalitions per state (G = 11 -> 2048), cheap for small MLPs.
    Absent groups are set to the baseline ("free space, no goal information, no motion").
    Returns (N, G) values that sum to f(x) - f(baseline).
    """
    X, T, B = _prep(X, targets, baseline)
    names = list(groups)
    G = len(names)
    N, D = X.shape
    masks = np.array(list(itertools.product([0, 1], repeat=G)), dtype=bool)  # (2^G, G)
    feat_mask = np.zeros((len(masks), D), dtype=np.float32)
    for g, name in enumerate(names):
        feat_mask[:, groups[name]] = masks[:, [g]]
    feat_mask = torch.as_tensor(feat_mask)
    size = masks.sum(1)
    w = np.array([factorial(s) * factorial(G - s - 1) / factorial(G) for s in range(G)])
    idx = {tuple(m): i for i, m in enumerate(masks)}
    with_i, without_i, weight = [], [], []
    for g in range(G):
        sel = np.where(~masks[:, g])[0]
        m_with = masks[sel].copy()
        m_with[:, g] = True
        with_i.append([idx[tuple(m)] for m in m_with])
        without_i.append(sel)
        weight.append(w[size[sel]])
    out = np.zeros((N, G), dtype=np.float32)
    for s in range(0, N, batch):
        xb, bb, tb = X[s:s + batch], B[s:s + batch], T[s:s + batch]
        nb = len(xb)
        inp = bb[:, None, :] + feat_mask[None] * (xb - bb)[:, None, :]  # (nb, 2^G, D)
        v = _target_value(model, inp.reshape(-1, D), tb.repeat_interleave(len(masks)), prob=True)
        v = v.reshape(nb, len(masks)).numpy()
        for g in range(G):
            out[s:s + nb, g] = ((v[:, with_i[g]] - v[:, without_i[g]]) * weight[g]).sum(1)
    return out


METHODS = {
    "gradient": gradient,
    "grad_x_input": grad_x_input,
    "integrated_grads": integrated_grads,
    "smoothgrad": smoothgrad,
    "occlusion": occlusion,
}


def attribute(model, X, targets, method: str = "integrated_grads", baseline=None, **kw) -> np.ndarray:
    return METHODS[method](model, X, targets, baseline=baseline, **kw)


def group_reduce(attr: np.ndarray, groups: dict[str, list[int]], how: str = "sum") -> np.ndarray:
    """Collapse per-feature attributions (N, D) into per-group attributions (N, G)."""
    f = np.sum if how == "sum" else np.mean
    return np.stack([f(attr[:, idx], axis=1) for idx in groups.values()], 1)
