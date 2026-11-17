"""Per-decision explanations and automatic mining of representative / unexpected decisions."""
from __future__ import annotations

import numpy as np

from ..envs.nav_env import NavEnv
from .saliency import attribute, group_shapley


class DecisionExplainer:
    """Bundles every explanation of a single state for one trained policy."""

    def __init__(self, model, env: NavEnv):
        self.model, self.env = model, env
        self.groups = env.feature_groups()
        self.group_names = list(self.groups)
        self.feature_names = env.feature_names()
        self.action_names = env.action_names()
        self.n_rays, self.max_range = env.n_rays, env.max_range
        self.baseline = env.baseline_obs()

    def explain(self, obs: np.ndarray, action=None, ig_steps: int = 32) -> dict:
        X = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        probs = self.model.probs_np(X)[0]
        if self.model.continuous:
            target = 1  # explain the steering command
            act_label = f"v={probs[0]:+.2f}, w={probs[1]:+.2f}"
        else:
            target = int(np.argmax(probs)) if action is None else int(action)
            act_label = self.action_names[target]
        T = np.array([target])
        ig = attribute(self.model, X, T, "integrated_grads", self.baseline, steps=ig_steps)[0]
        grad = attribute(self.model, X, T, "gradient", self.baseline)[0]
        shap = group_shapley(self.model, X, T, self.groups, self.baseline)[0]
        att = self.model.attention_np(X)
        ent = float(-(probs * np.log(probs + 1e-9)).sum()) if not self.model.continuous else float("nan")
        out = {"action": target, "action_name": act_label, "probs": probs.tolist(), "entropy": ent,
               "ig": ig.tolist(), "gradient": grad.tolist(), "shapley": dict(zip(self.group_names, shap.tolist())),
               "attention": att[0].tolist() if att is not None else None}
        out["text"] = self.to_text(X[0], out)
        return out

    def _group_context(self, x, name):
        if name.startswith("lidar:"):
            d = x[self.groups[name]].min() * self.max_range
            sector = name.split(":")[1]
            return f"{sector} sector ({'obstacle at %.1f m' % d if d < self.max_range * 0.98 else 'clear'})"
        n = self.n_rays
        if name == "goal_heading":
            ang = np.degrees(np.arctan2(x[n + 1], x[n + 2]))
            side = "left" if ang > 0 else "right"
            return f"goal direction ({abs(ang):.0f} deg to the {side})" if abs(ang) > 3 else "goal direction (straight ahead)"
        if name == "goal_distance":
            return f"goal distance ({x[n] * self.env.diag:.1f} m)"
        return f"current motion (v={x[n + 3] * self.env.v_max:.2f} m/s, w={x[n + 4] * self.env.w_max:+.1f} rad/s)"

    def to_text(self, x, e) -> str:
        shap = e["shapley"]
        top = sorted(shap.items(), key=lambda kv: -abs(kv[1]))[:3]
        conf = e["probs"][e["action"]] if not self.model.continuous else None
        head = f"Chose '{e['action_name']}'" + (f" (p={conf:.2f})" if conf is not None else "") + ". "
        parts = []
        for name, v in top:
            if abs(v) < 0.01:
                continue
            verb = "supported" if v > 0 else "argued against"
            parts.append(f"the {self._group_context(x, name)} {verb} it ({v:+.2f})")
        if not parts:
            return head + "No single input group dominated this decision."
        return head + "Main factors: " + "; ".join(parts) + "."


def flag_decisions(rec: dict, model, env: NavEnv, near_miss_m: float = 0.35, entropy_frac: float = 0.6,
                   window: int = 6) -> list[dict]:
    """Flags steps worth reviewing in a recorded episode (see ``eval.metrics.run_episode(record=True)``).

    Categories
        near_miss       : closest lidar return below ``near_miss_m``
        uncertain       : policy entropy above ``entropy_frac`` * log(|A|)
        away_from_goal  : goal within 30 deg ahead, front clear, but the agent turns hard / spins
        oscillation     : left and right turns alternating within a short window
        pre_failure     : the last ``window`` steps before a collision or timeout
    """
    X = np.asarray(rec["obs"], dtype=np.float32)
    acts = np.asarray(rec["actions"])
    n = env.n_rays
    flags = []
    probs = model.probs_np(X)
    for t in range(len(acts)):
        x, a = X[t], acts[t]
        reasons = []
        min_d = x[:n].min() * env.max_range
        if min_d < near_miss_m:
            reasons.append("near_miss")
        if not model.continuous:
            p_ = probs[t]
            ent = -(p_ * np.log(p_ + 1e-9)).sum()
            if ent > entropy_frac * np.log(len(p_)):
                reasons.append("uncertain")
            ang = np.degrees(np.arctan2(x[n + 1], x[n + 2]))
            front = x[[0, 1, n - 1]].min() * env.max_range
            if abs(ang) < 30 and front > 1.5 and int(a) in (3, 4):
                reasons.append("away_from_goal")
            lo, hi = max(0, t - window), t + 1
            seq = [int(v) for v in acts[lo:hi]]
            lefts = sum(v in (1, 3) for v in seq)
            rights = sum(v in (2, 4) for v in seq)
            changes = sum(1 for i in range(1, len(seq)) if (seq[i] in (1, 3) and seq[i - 1] in (2, 4)) or
                          (seq[i] in (2, 4) and seq[i - 1] in (1, 3)))
            if lefts >= 2 and rights >= 2 and changes >= 3:
                reasons.append("oscillation")
        if not rec["metrics"]["success"] and t >= len(acts) - window:
            reasons.append("pre_failure")
        if reasons:
            flags.append({"t": t, "reasons": reasons, "min_dist": float(min_d)})
    return flags


def select_representative(flags: list[dict], max_items: int = 8) -> list[dict]:
    """Pick a diverse handful: at most two per category, earliest first."""
    picked, per_cat = [], {}
    for f in flags:
        cat = f["reasons"][0]
        if per_cat.get(cat, 0) < 2 and all(abs(f["t"] - p["t"]) > 3 for p in picked):
            picked.append(f)
            per_cat[cat] = per_cat.get(cat, 0) + 1
        if len(picked) >= max_items:
            break
    return picked
