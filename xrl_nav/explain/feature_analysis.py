"""Global, feature-level analysis of a trained navigation policy.

* ``collect_states``         : roll out the policy and build a dataset of visited states
* ``engineered_features``    : human-level features (sector min distances, heading error in degrees, ...)
* ``global_importance``      : mean |attribution| per feature / group, overall and per chosen action
* ``action_conditioned_stats``: what the world looked like when each action was chosen
* ``partial_dependence``     : how action probabilities change as one feature is swept
* ``permutation_importance`` : model-agnostic: action agreement drop when a group is shuffled
* ``surrogate_tree``         : depth-limited decision tree distilled from the policy, with fidelity
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..envs.nav_env import SECTOR_NAMES, NavEnv
from .saliency import attribute, group_reduce


def collect_states(model, env_kwargs: dict, n_episodes: int = 30, split: str = "train", seed0: int = 5_000,
                   scenarios: list[str] | None = None) -> dict:
    env = NavEnv(**{**env_kwargs, "layout_mode": split})
    obs_l, act_l, ep_l, step_l, pose_l, outcome = [], [], [], [], [], []
    runs = [(None, seed0 + i) for i in range(n_episodes)] + [(s, 0) for s in (scenarios or [])]
    for ep, (scen, seed) in enumerate(runs):
        obs, _ = env.reset(seed=seed, options={"scenario": scen} if scen else None)
        done, t, ep_obs = False, 0, []
        while not done:
            a = model.act(obs)
            obs_l.append(obs)
            act_l.append(np.asarray(a))
            ep_l.append(ep)
            step_l.append(t)
            pose_l.append(env.pose.copy())
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
            t += 1
        outcome += ["success" if info["success"] else "collision" if info["collision"] else "timeout"] * t
    env.close()
    names = NavEnv(**env_kwargs).feature_names()
    return {"X": np.array(obs_l, dtype=np.float32), "actions": np.array(act_l), "episode": np.array(ep_l),
            "step": np.array(step_l), "pose": np.array(pose_l), "outcome": np.array(outcome),
            "feature_names": names}


def engineered_features(X: np.ndarray, n_rays: int = 24, max_range: float = 4.0) -> pd.DataFrame:
    per = n_rays // 8
    cols = {}
    for k, name in enumerate(SECTOR_NAMES):
        idx = [(k * per + j) % n_rays for j in range(-(per // 2), per - per // 2)]
        cols[f"min_dist_{name}"] = X[:, idx].min(1) * max_range
    cols["goal_heading_deg"] = np.degrees(np.arctan2(X[:, n_rays + 1], X[:, n_rays + 2]))
    cols["goal_dist"] = X[:, n_rays]
    cols["lin_vel"] = X[:, n_rays + 3]
    cols["ang_vel"] = X[:, n_rays + 4]
    return pd.DataFrame(cols)


def global_importance(model, X, actions, groups, baseline, method="integrated_grads", feature_names=None):
    targets = actions if not model.continuous else np.zeros(len(X), dtype=int)
    A = attribute(model, X, targets, method=method, baseline=baseline)
    feat = pd.Series(np.abs(A).mean(0), index=feature_names).sort_values(ascending=False)
    G = group_reduce(A, groups)
    grp = pd.DataFrame(np.abs(G), columns=list(groups))
    grp_overall = grp.mean().sort_values(ascending=False)
    per_action = None
    if not model.continuous:
        grp["action"] = actions
        per_action = grp.groupby("action").mean()
        signed = pd.DataFrame(G, columns=list(groups))
        signed["action"] = actions
        per_action_signed = signed.groupby("action").mean()
    else:
        per_action_signed = None
    return {"attr": A, "feature": feat, "group": grp_overall, "per_action": per_action,
            "per_action_signed": per_action_signed}


def action_conditioned_stats(X, actions, action_names, n_rays=24, max_range=4.0) -> pd.DataFrame:
    ef = engineered_features(X, n_rays, max_range)
    ef["action"] = [action_names[int(a)] for a in actions]
    stats = ef.groupby("action").mean()
    stats["count"] = ef.groupby("action").size()
    stats["share"] = stats["count"] / len(ef)
    return stats


def partial_dependence(model, X, feature_idx: list[int], grid: np.ndarray, transform=None) -> np.ndarray:
    """Average model output (probs for discrete, mean action for continuous) as features are swept.

    ``transform(X_copy, value)`` can set several coupled features at once (e.g. goal sin/cos from an angle).
    Returns (len(grid), n_outputs).
    """
    out = []
    for v in grid:
        Xc = X.copy()
        if transform is not None:
            transform(Xc, v)
        else:
            Xc[:, feature_idx] = v
        out.append(model.probs_np(Xc).mean(0))
    return np.array(out)


def permutation_importance(model, X, groups, n_repeats: int = 5, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    base = model.probs_np(X)
    base_a = base.argmax(1) if not model.continuous else base
    res = {}
    for name, idx in groups.items():
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[:, idx] = X[perm][:, idx]
            out = model.probs_np(Xp)
            if model.continuous:
                drops.append(float(np.abs(out - base_a).mean()))
            else:
                drops.append(float(1.0 - (out.argmax(1) == base_a).mean()))
        res[name] = float(np.mean(drops))
    return pd.Series(res).sort_values(ascending=False)


def surrogate_tree(model, X, max_depth: int = 4, n_rays: int = 24, max_range: float = 4.0, action_names=None,
                   seed: int = 0):
    """Distil the policy into a shallow decision tree over engineered features. Discrete actions only."""
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier, export_text

    y = model.probs_np(X).argmax(1)
    F = engineered_features(X, n_rays, max_range)
    Xtr, Xte, ytr, yte = train_test_split(F, y, test_size=0.3, random_state=seed)
    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=20, random_state=seed).fit(Xtr, ytr)
    from sklearn.metrics import balanced_accuracy_score

    pred = tree.predict(Xte)
    fidelity = float((pred == yte).mean())
    balanced = float(balanced_accuracy_score(yte, pred))
    majority = float((yte == np.bincount(ytr).argmax()).mean())
    names = [action_names[c] for c in tree.classes_] if action_names else None
    rules = export_text(tree, feature_names=list(F.columns), class_names=names, decimals=2)
    return {"tree": tree, "fidelity": fidelity, "balanced_fidelity": balanced, "majority_baseline": majority, "rules": rules,
            "feature_importance": pd.Series(tree.feature_importances_, index=F.columns).sort_values(ascending=False)}
