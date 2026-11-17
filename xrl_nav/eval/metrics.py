"""Navigation performance evaluation.

Metrics per episode
    success          : reached the goal radius before collision / timeout
    collision        : episode ended in (or contained) a collision
    reward           : undiscounted episode return
    steps            : episode length
    path_length      : distance actually travelled (m)
    optimal_length   : geodesic shortest path on an occupancy grid inflated by the robot radius (m)
    path_efficiency  : optimal_length / path_length for successful episodes (1.0 = optimal), NaN otherwise
    min_clearance    : closest lidar return during the episode (m)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..envs.geometry import shortest_path_length
from ..envs.layouts import ROBOT_RADIUS
from ..envs.nav_env import NavEnv


def run_episode(env: NavEnv, policy, seed: int | None = None, options: dict | None = None,
                record: bool = False, deterministic: bool = True):
    obs, info = env.reset(seed=seed, options=options)
    layout = env.layout
    total, done = 0.0, False
    rec = {"obs": [obs], "actions": [], "rewards": [], "poses": [list(env.pose)], "rays": [env._last_rays.copy()]}
    while not done:
        a = policy.act(obs, deterministic=deterministic)
        obs, r, term, trunc, info = env.step(a)
        total += r
        done = term or trunc
        if record:
            rec["actions"].append(np.asarray(a).tolist())
            rec["rewards"].append(r)
            rec["obs"].append(obs)
            rec["poses"].append(list(env.pose))
            rec["rays"].append(env._last_rays.copy())
    opt = shortest_path_length(layout["obstacles"], env.arena_half, ROBOT_RADIUS, layout["start"], layout["goal"])
    opt = max(opt - env.goal_radius, 1e-6)  # the agent only needs to enter the goal radius
    eff = min(opt / info["path_length"], 1.0) if info["success"] and info["path_length"] > 0 else np.nan
    row = {"seed": seed, "layout": layout["name"], "success": int(info["success"]),
           "collision": int(info["collision"]), "timeout": int(not info["success"] and not info["collision"]),
           "reward": total, "steps": info["steps"], "path_length": info["path_length"],
           "optimal_length": opt, "path_efficiency": eff, "min_clearance": info["min_clearance"]}
    if record:
        rec["obs"] = rec["obs"][:-1] if len(rec["obs"]) > len(rec["actions"]) else rec["obs"]
        rec.update({"layout": layout, "metrics": row})
        return row, rec
    return row, None


def evaluate(policy, env_kwargs: dict, n_episodes: int = 100, seed0: int = 1_000, split: str = "train",
             scenario: str | None = None) -> pd.DataFrame:
    env = NavEnv(**{**env_kwargs, "layout_mode": split, "scenario_name": scenario})
    rows = []
    for i in range(n_episodes):
        row, _ = run_episode(env, policy, seed=seed0 + i)
        row["split"] = scenario or split
        rows.append(row)
    env.close()
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> dict:
    def ci(x):
        x = np.asarray(x, dtype=float)
        x = x[~np.isnan(x)]
        if len(x) < 2:
            return float(np.mean(x)) if len(x) else float("nan"), float("nan")
        return float(np.mean(x)), float(1.96 * np.std(x, ddof=1) / np.sqrt(len(x)))

    out = {"episodes": len(df)}
    for k in ["success", "collision", "timeout", "reward", "steps", "path_efficiency", "min_clearance"]:
        m, h = ci(df[k])
        out[k] = m
        out[k + "_ci95"] = h
    return out
