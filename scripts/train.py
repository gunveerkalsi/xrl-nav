#!/usr/bin/env python3
"""Train one agent.

Examples
    python scripts/train.py --algo ppo --policy mlp --seed 0
    python scripts/train.py --algo ppo --policy attention --timesteps 500000
    python scripts/train.py --algo dqn --timesteps 300000
    python scripts/train.py --algo ppo --continuous
    python scripts/train.py --experiment ppo_mlp            # use settings from configs/default.yaml
"""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xrl_nav.agents.train import train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--experiment", help="experiment label from the config (runs all its seeds)")
    ap.add_argument("--algo", default="ppo", choices=["ppo", "dqn"])
    ap.add_argument("--policy", default="mlp", choices=["mlp", "attention"])
    ap.add_argument("--continuous", action="store_true")
    ap.add_argument("--timesteps", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--subproc", action="store_true", help="run envs in subprocesses")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    env_kwargs = cfg["env"]

    if args.experiment:
        ex = cfg["experiments"][args.experiment]
        for s in ex["seeds"]:
            out = os.path.join("runs", f"{args.experiment}_s{s}")
            print(f"== {args.experiment} seed {s} -> {out}")
            train(ex["algo"], ex["policy"], ex["continuous"], ex["timesteps"], s, ex["n_envs"], out, env_kwargs,
                  subproc=args.subproc)
        return
    label = f"{args.algo}_{args.policy}" + ("_cont" if args.continuous else "")
    out = args.out or os.path.join("runs", f"{label}_s{args.seed}")
    train(args.algo, args.policy, args.continuous, args.timesteps, args.seed, args.n_envs, out, env_kwargs,
          subproc=args.subproc)


if __name__ == "__main__":
    main()
