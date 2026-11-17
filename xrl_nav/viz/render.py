"""Top-down rendering of the agent together with its explanations (frames, GIF, MP4)."""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patches, transforms  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from ..envs.layouts import ROBOT_RADIUS  # noqa: E402

SUPPORT, AGAINST, NEUTRAL = "#c0392b", "#2471a3", "#bbbbbb"


def draw_obstacles(ax, layout, arena_half):
    ax.add_patch(patches.Rectangle((-arena_half, -arena_half), 2 * arena_half, 2 * arena_half, fill=False, lw=2,
                                   ec="#333"))
    for ob in layout["obstacles"]:
        if ob["type"] == "cylinder":
            ax.add_patch(patches.Circle(ob["pos"], ob["radius"], fc="#d68a3c", ec="#7a4a1a"))
        else:
            hx, hy = ob["half"]
            r = patches.Rectangle((-hx, -hy), 2 * hx, 2 * hy, fc="#c0703a", ec="#7a4a1a")
            r.set_transform(transforms.Affine2D().rotate(ob["yaw"]).translate(*ob["pos"]) + ax.transData)
            ax.add_patch(r)
    ax.add_patch(patches.Circle(layout["goal"], 0.35, fc="#2ecc71", alpha=0.5, ec="#1e8449"))
    ax.plot(*layout["start"], marker="s", color="#555", ms=6)
    ax.set_xlim(-arena_half - 0.2, arena_half + 0.2)
    ax.set_ylim(-arena_half - 0.2, arena_half + 0.2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def draw_robot(ax, pose, rays=None, max_range=4.0, ray_weights=None, signed=True, trajectory=None):
    x, y, th = pose
    if trajectory is not None and len(trajectory) > 1:
        tr = np.asarray(trajectory)
        ax.plot(tr[:, 0], tr[:, 1], color="#1f618d", lw=1.5, alpha=0.7)
    if rays is not None:
        n = len(rays)
        ang = th + np.arange(n) * 2 * np.pi / n
        L = ROBOT_RADIUS + np.asarray(rays) * max_range
        segs = [[(x, y), (x + L[i] * np.cos(ang[i]), y + L[i] * np.sin(ang[i]))] for i in range(n)]
        if ray_weights is None:
            colors, widths = [NEUTRAL] * n, [0.8] * n
        else:
            w = np.asarray(ray_weights, dtype=float)
            m = np.abs(w).max() + 1e-9
            colors = [(SUPPORT if v > 0 else AGAINST) if signed else "#8e44ad" for v in w]
            widths = 0.5 + 4.0 * np.abs(w) / m
            alphas = 0.25 + 0.75 * np.abs(w) / m
            colors = [matplotlib.colors.to_rgba(c, a) for c, a in zip(colors, alphas)]
        ax.add_collection(LineCollection(segs, colors=colors, linewidths=widths, zorder=2))
        hits = [(s[1], r) for s, r in zip(segs, rays) if r < 0.98]
        if hits:
            ax.scatter([h[0][0] for h in hits], [h[0][1] for h in hits], s=6, c="k", zorder=3)
    ax.add_patch(patches.Circle((x, y), ROBOT_RADIUS, fc="#2e86c1", ec="k", zorder=4))
    ax.arrow(x, y, 0.35 * np.cos(th), 0.35 * np.sin(th), width=0.04, color="w", zorder=5, length_includes_head=True)


def explanation_frame(layout, arena_half, max_range, pose, rays, trajectory, expl, action_names, group_names,
                      step: int, title: str = "", ray_mode: str = "ig", n_rays: int = 24, flag=None):
    """One composite frame: scene with saliency-coloured lidar + action probs + group Shapley + polar view."""
    fig = plt.figure(figsize=(12, 6.4), dpi=80)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1], height_ratios=[1, 1.4, 1.3], hspace=0.95, wspace=0.18)
    ax = fig.add_subplot(gs[:, 0])
    draw_obstacles(ax, layout, arena_half)
    if ray_mode == "attention" and expl.get("attention") is not None:
        w, signed, label = expl["attention"], False, "lidar coloured by attention"
    else:
        w, signed, label = expl["ig"][:n_rays], True, "lidar: red = supports action, blue = against (IG)"
    draw_robot(ax, pose, rays, max_range, w, signed, trajectory)
    ax.set_title(f"{title}  step {step}" + (f"   [{', '.join(flag)}]" if flag else ""), fontsize=11)
    ax.text(0.01, -0.03, label, transform=ax.transAxes, fontsize=8, va="top", color="#444")

    a1 = fig.add_subplot(gs[0, 1])
    probs = np.asarray(expl["probs"])
    cols = ["#7f8c8d"] * len(probs)
    if len(probs) == len(action_names) and len(action_names) > 2:
        cols[expl["action"]] = "#e67e22"
        a1.bar(action_names, probs, color=cols)
        a1.set_ylim(0, 1)
        a1.set_title("policy  pi(a | s)", fontsize=10)
    else:
        a1.bar(["v", "w"], probs, color=["#7f8c8d", "#e67e22"])
        a1.set_ylim(-1, 1)
        a1.set_title("action mean", fontsize=10)
    a1.tick_params(labelsize=8)

    a2 = fig.add_subplot(gs[1, 1])
    vals = np.array([expl["shapley"][g] for g in group_names])
    order = np.argsort(np.abs(vals))
    a2.barh([group_names[i].replace("lidar:", "") for i in order], vals[order],
            color=[SUPPORT if vals[i] > 0 else AGAINST for i in order])
    a2.axvline(0, color="k", lw=0.6)
    a2.set_title("group Shapley contribution to chosen action", fontsize=10)
    a2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    a2.tick_params(labelsize=8)

    a3 = fig.add_subplot(gs[2, 1], projection="polar")
    ang = np.arange(n_rays) * 2 * np.pi / n_rays
    ig = np.abs(np.asarray(expl["ig"][:n_rays]))
    a3.bar(ang, ig / (ig.max() + 1e-9), width=2 * np.pi / n_rays * 0.8, color="#c0392b", alpha=0.6, label="|IG|")
    if expl.get("attention") is not None:
        at = np.asarray(expl["attention"])
        a3.plot(np.r_[ang, ang[:1]], np.r_[at, at[:1]] / (at.max() + 1e-9), color="#8e44ad", lw=1.5,
                label="attention")
    a3.set_theta_zero_location("N")
    a3.set_yticks([])
    a3.set_xticks(np.radians([0, 90, 180, 270]))
    a3.set_xticklabels(["front", "left", "back", "right"], fontsize=7)
    a3.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.6, 1.15))
    a3.set_title("per-beam importance (robot frame)", fontsize=9, pad=6)

    fig.text(0.02, 0.01, _wrap(expl["text"], 150), fontsize=9, va="bottom", family="monospace")
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img


def _wrap(s, width):
    import textwrap

    return "\n".join(textwrap.wrap(s, width))


def write_video(frames, path_no_ext: str, fps: int = 8, gif: bool = True, mp4: bool = True):
    import imageio.v2 as imageio

    out = []
    if mp4:
        try:
            imageio.mimsave(path_no_ext + ".mp4", frames, fps=fps, macro_block_size=8)
            out.append(path_no_ext + ".mp4")
        except Exception as e:  # ffmpeg missing
            print("mp4 export skipped:", e)
    if gif:
        imageio.mimsave(path_no_ext + ".gif", frames[::2], duration=2 / fps, loop=0)
        out.append(path_no_ext + ".gif")
    return out


def record_explained_episode(model, env, explainer, seed=None, scenario=None, flags_fn=None, max_steps=None):
    """Runs one episode and returns a JSON-serialisable record with per-step explanations."""
    from ..eval.metrics import run_episode

    opts = {"scenario": scenario} if scenario else None
    row, rec = run_episode(env, model, seed=seed, options=opts, record=True)
    steps = []
    T = len(rec["actions"]) if max_steps is None else min(max_steps, len(rec["actions"]))
    for t in range(T):
        e = explainer.explain(rec["obs"][t], rec["actions"][t] if not model.continuous else None)
        steps.append({"t": t, "pose": [float(v) for v in rec["poses"][t]],
                      "rays": [float(v) for v in rec["rays"][t]], **e})
    flags = flags_fn(rec) if flags_fn else []
    return {"layout": rec["layout"], "metrics": {k: (None if isinstance(v, float) and np.isnan(v) else v)
                                                 for k, v in row.items()},
            "steps": steps, "flags": flags, "group_names": explainer.group_names,
            "action_names": explainer.action_names, "arena_half": env.arena_half, "max_range": env.max_range,
            "n_rays": env.n_rays, "final_pose": [float(v) for v in rec["poses"][-1]]}


def render_episode(ep: dict, out_no_ext: str, title: str = "", ray_mode: str = "ig", fps: int = 8, every: int = 1):
    frames = []
    flag_at = {f["t"]: f["reasons"] for f in ep["flags"]}
    traj = []
    for s in ep["steps"]:
        traj.append(s["pose"][:2])
        if s["t"] % every:
            continue
        frames.append(explanation_frame(ep["layout"], ep["arena_half"], ep["max_range"], s["pose"], s["rays"],
                                        traj, s, ep["action_names"], ep["group_names"], s["t"], title, ray_mode,
                                        ep["n_rays"], flag_at.get(s["t"])))
    frames += [frames[-1]] * fps  # hold last frame
    return write_video(frames, out_no_ext, fps=fps)


def save_episode_json(ep: dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(ep, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
