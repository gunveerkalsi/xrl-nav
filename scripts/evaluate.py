#!/usr/bin/env python3
"""Evaluate trained agents and baselines on training layouts, held-out layouts and named scenarios.

    python scripts/evaluate.py                       # every runs/*/model.zip + baselines
    python scripts/evaluate.py --episodes 50 --runs runs/ppo_mlp_s0

Outputs (results/evaluation/)
    episodes.csv, summary.csv, summary.md, metrics.png, training_curves.png, trajectories_<agent>.png
"""
import argparse
import glob
import json
import os
import re
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xrl_nav.agents.baselines import PotentialFieldPolicy, RandomPolicy  # noqa: E402
from xrl_nav.agents.policies import load_policy  # noqa: E402
from xrl_nav.envs.layouts import SCENARIOS  # noqa: E402
from xrl_nav.envs.nav_env import NavEnv  # noqa: E402
from xrl_nav.eval.metrics import evaluate, run_episode, summarise  # noqa: E402
from xrl_nav.viz import plots  # noqa: E402


def agent_label(run_dir):
    return re.sub(r"_s\d+$", "", os.path.basename(run_dir.rstrip("/")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--out", default="results/evaluation")
    ap.add_argument("--no-baselines", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    env_kwargs = cfg["env"]
    n_ep = args.episodes or cfg["evaluation"]["episodes"]
    n_scen = cfg["evaluation"]["scenario_episodes"]
    os.makedirs(args.out, exist_ok=True)
    runs = args.runs or sorted(d for d in glob.glob("runs/*") if os.path.exists(f"{d}/model.zip"))

    agents = []  # (label, seed, policy, continuous)
    for d in runs:
        pm = load_policy(f"{d}/model.zip")
        agents.append((agent_label(d), pm.meta.get("seed", 0), pm, pm.continuous))
    if not args.no_baselines:
        tmp = NavEnv(**env_kwargs)
        agents.append(("random", 0, RandomPolicy(tmp.action_space), False))
        agents.append(("potential_field", 0, PotentialFieldPolicy(env_kwargs["n_rays"], env_kwargs["max_range"]),
                       False))
        tmp.close()

    frames = []
    for label, seed, pol, cont in agents:
        ek = {**env_kwargs, "continuous": cont}
        for split in ("train", "heldout"):
            df = evaluate(pol, ek, n_ep, seed0=20_000, split=split)
            df["agent"], df["train_seed"] = label, seed
            frames.append(df)
        for sc in SCENARIOS:
            df = evaluate(pol, ek, n_scen, seed0=30_000, scenario=sc)
            df["agent"], df["train_seed"] = label, seed
            frames.append(df)
        print(f"evaluated {label} (seed {seed})")
    allep = pd.concat(frames, ignore_index=True)
    allep.to_csv(f"{args.out}/episodes.csv", index=False)

    rows = []
    for (agent, split), g in allep.groupby(["agent", "split"], sort=False):
        s = summarise(g)
        s.update({"agent": agent, "split": split, "n_seeds": g.train_seed.nunique()})
        rows.append(s)
    summ = pd.DataFrame(rows)
    summ.to_csv(f"{args.out}/summary.csv", index=False)
    cols = ["agent", "split", "n_seeds", "episodes", "success", "collision", "timeout", "reward",
            "path_efficiency", "min_clearance"]
    with open(f"{args.out}/summary.md", "w") as f:
        f.write(summ[cols].round(3).to_markdown(index=False))
    print(summ[cols].round(3).to_string(index=False))

    main_splits = summ[summ.split.isin(["train", "heldout"])]
    plots.metrics_bars(main_splits, f"{args.out}/metrics.png")
    scen = summ[summ.split.isin(SCENARIOS)].copy()
    if len(scen):
        piv = scen.pivot(index="agent", columns="split", values="success")[SCENARIOS]
        piv.to_csv(f"{args.out}/scenario_success.csv")
        with open(f"{args.out}/scenario_success.md", "w") as f:
            f.write(piv.round(2).to_markdown())

    curve_runs = {os.path.basename(d): d for d in runs if os.path.exists(f"{d}/episodes.csv")}
    if curve_runs:
        plots.training_curves(curve_runs, f"{args.out}/training_curves.png")

    # one trajectory overview per agent type (first seed), on every scenario
    seen = set()
    for label, seed, pol, cont in agents:
        if label in seen:
            continue
        seen.add(label)
        env = NavEnv(**{**env_kwargs, "continuous": cont})
        eps = []
        for sc in SCENARIOS + ["heldout"]:
            opts = {"scenario": sc} if sc != "heldout" else {"layout_mode": "heldout"}
            row, rec = run_episode(env, pol, seed=7, options=opts, record=True)
            eps.append({"layout": rec["layout"], "arena_half": env.arena_half, "metrics": row,
                        "steps": [{"pose": p} for p in rec["poses"][:-1]], "final_pose": rec["poses"][-1]})
        env.close()
        for e in eps:
            if e["metrics"]["path_efficiency"] != e["metrics"]["path_efficiency"]:
                e["metrics"]["path_efficiency"] = None
        plots.trajectories_grid(eps, f"{args.out}/trajectories_{label}.png")
    json.dump({"runs": runs, "episodes_per_split": n_ep}, open(f"{args.out}/eval_meta.json", "w"), indent=2)


if __name__ == "__main__":
    main()
