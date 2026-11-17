"""Static figures for the report."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .render import draw_obstacles, draw_robot  # noqa: E402

PALETTE = ["#2e86c1", "#e67e22", "#27ae60", "#8e44ad", "#c0392b", "#7f8c8d", "#16a085"]


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def training_curves(runs: dict[str, str], path: str, window: int = 100):
    """runs: label -> run directory containing episodes.csv and evaluations.csv"""
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for i, (label, d) in enumerate(runs.items()):
        c = PALETTE[i % len(PALETTE)]
        ep = pd.read_csv(f"{d}/episodes.csv")
        axs[0].plot(ep.timestep, ep.reward.rolling(window, min_periods=10).mean(), color=c, label=label)
        axs[1].plot(ep.timestep, ep.success.rolling(window, min_periods=10).mean(), color=c, label=label)
        ev = pd.read_csv(f"{d}/evaluations.csv")
        for split, ls in (("train", "-"), ("heldout", "--")):
            e = ev[ev.split == split]
            axs[2].plot(e.timestep, e.success_rate, ls, color=c, marker="o", ms=3,
                        label=f"{label} ({split})")
    axs[0].set_title("episode reward (rolling mean)")
    axs[1].set_title("training success rate (rolling)")
    axs[2].set_title("deterministic eval success: train (solid) vs held-out (dashed)")
    for a in axs:
        a.set_xlabel("environment steps")
        a.grid(alpha=0.3)
    axs[0].legend(fontsize=8)
    axs[2].legend(fontsize=7)
    return _save(fig, path)


def metrics_bars(summary: pd.DataFrame, path: str):
    """summary: rows = agent, columns incl. split, success, collision, path_efficiency (+ _ci95)."""
    metrics = [("success", "success rate"), ("collision", "collision rate"), ("path_efficiency", "path efficiency"),
               ("reward", "episode reward")]
    fig, axs = plt.subplots(1, 4, figsize=(17, 4))
    splits = list(summary.split.unique())
    agents = list(summary.agent.unique())
    w = 0.8 / len(splits)
    for ax, (m, title) in zip(axs, metrics):
        for j, sp in enumerate(splits):
            sub = summary[summary.split == sp].set_index("agent").reindex(agents)
            ax.bar(np.arange(len(agents)) + j * w, sub[m], w, yerr=sub.get(m + "_ci95"), capsize=2,
                   color=PALETTE[j % len(PALETTE)], label=sp)
        ax.set_xticks(np.arange(len(agents)) + w * (len(splits) - 1) / 2)
        ax.set_xticklabels(agents, rotation=30, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axs[0].legend(fontsize=8)
    return _save(fig, path)


def trajectories_grid(episodes: list[dict], path: str, cols: int = 3):
    rows = int(np.ceil(len(episodes) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows))
    axs = np.atleast_1d(axs).ravel()
    for ax, ep in zip(axs, episodes):
        draw_obstacles(ax, ep["layout"], ep["arena_half"])
        traj = [s["pose"][:2] for s in ep["steps"]] + [ep["final_pose"][:2]]
        draw_robot(ax, ep["final_pose"], trajectory=traj)
        m = ep["metrics"]
        res = "success" if m["success"] else "collision" if m["collision"] else "timeout"
        eff = m.get("path_efficiency")
        ax.set_title(f"{ep['layout']['name']}: {res}" + (f", eff={eff:.2f}" if eff else ""), fontsize=10)
    for ax in axs[len(episodes):]:
        ax.axis("off")
    return _save(fig, path)


def importance_bars(series_by_method: dict[str, pd.Series], path: str, title: str):
    names = list(next(iter(series_by_method.values())).index)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    w = 0.8 / len(series_by_method)
    for j, (m, s) in enumerate(series_by_method.items()):
        v = s.reindex(names).values
        ax.bar(np.arange(len(names)) + j * w, v / (np.nanmax(v) + 1e-12), w, label=m, color=PALETTE[j])
    ax.set_xticks(np.arange(len(names)) + w * (len(series_by_method) - 1) / 2)
    ax.set_xticklabels([n.replace("lidar:", "") for n in names], rotation=35, ha="right")
    ax.set_ylabel("normalised importance")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, path)


def per_action_heatmap(df: pd.DataFrame, action_names, path: str, title: str):
    fig, ax = plt.subplots(figsize=(11, 3.6))
    vals = df.values
    lim = np.abs(vals).max()
    im = ax.imshow(vals, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([action_names[int(i)] for i in df.index])
    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels([c.replace("lidar:", "") for c in df.columns], rotation=35, ha="right")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, f"{vals[i, j]:+.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.02)
    ax.set_title(title)
    return _save(fig, path)


def deletion_plot(curves: dict[str, pd.DataFrame], path: str, k_max: int = 15):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for i, (m, c) in enumerate(curves.items()):
        c = c[c.k <= k_max]
        style = dict(color="k", ls="--") if m == "random" else dict(color=PALETTE[i % len(PALETTE)])
        axs[0].plot(c.k, c.prob, marker="o", ms=3, label=m, **style)
        if "flip_rate" in c:
            axs[1].plot(c.k, c.flip_rate, marker="o", ms=3, label=m, **style)
    axs[0].set_title("probability of the original action as top-k features are removed")
    axs[1].set_title("action flip rate")
    for a in axs:
        a.set_xlabel("k features replaced by baseline")
        a.grid(alpha=0.3)
    axs[0].legend(fontsize=8)
    return _save(fig, path)


def agreement_heatmap(M: pd.DataFrame, path: str):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(M.values, cmap="viridis", vmin=-0.2, vmax=1)
    ax.set_xticks(range(len(M)))
    ax.set_xticklabels(M.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(M)))
    ax.set_yticklabels(M.index)
    for i in range(len(M)):
        for j in range(len(M)):
            ax.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
    fig.colorbar(im, ax=ax)
    ax.set_title("rank agreement between explanation methods")
    return _save(fig, path)


def pdp_plot(grid, curves, action_names, path, xlabel, title):
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, name in enumerate(action_names):
        ax.plot(grid, curves[:, i], label=name, color=PALETTE[i % len(PALETTE)])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("mean action probability")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def attention_vs_ig(ang_deg, att, ig, path):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection="polar")
    a = np.radians(np.r_[ang_deg, ang_deg[:1]])
    ax.plot(a, np.r_[att, att[:1]] / att.max(), label="mean attention", color="#8e44ad")
    ax.plot(a, np.r_[ig, ig[:1]] / ig.max(), label="mean |IG|", color="#c0392b")
    ax.set_theta_zero_location("N")
    ax.set_yticks([])
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("where the agent looks (robot frame, front = up)")
    return _save(fig, path)
