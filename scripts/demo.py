#!/usr/bin/env python3
"""Visual demonstration: the agent navigating together with its decision explanations.

    python scripts/demo.py --models runs/ppo_mlp_s0/model.zip runs/ppo_attention_s0/model.zip

Outputs (results/demo/)
    <run>_<scenario>.gif / .mp4   annotated videos (saliency-coloured lidar, policy, Shapley, text)
    replay.html                   interactive step-by-step decision review (open in a browser)
    episodes/<run>_<scenario>.json raw per-step explanations

    python scripts/demo.py --gui --models runs/ppo_mlp_s0/model.zip --scenario clutter
        opens the PyBullet 3D GUI (needs a display) and drives the agent live, printing explanations.
"""
import argparse
import os
import sys
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xrl_nav.agents.policies import load_policy  # noqa: E402
from xrl_nav.envs.layouts import SCENARIOS  # noqa: E402
from xrl_nav.envs.nav_env import NavEnv  # noqa: E402
from xrl_nav.explain.decisions import DecisionExplainer, flag_decisions  # noqa: E402
from xrl_nav.viz.render import record_explained_episode, render_episode, save_episode_json  # noqa: E402
from xrl_nav.viz.replay_html import build_replay_html  # noqa: E402


def gui_demo(model, env_kwargs, scenario):
    env = NavEnv(**{**env_kwargs, "continuous": model.continuous, "render_mode": "human"})
    ex = DecisionExplainer(model, env)
    obs, _ = env.reset(options={"scenario": scenario} if scenario else None)
    done = False
    while not done:
        e = ex.explain(obs)
        print(e["text"])
        obs, _, term, trunc, info = env.step(model.act(obs))
        done = term or trunc
        time.sleep(env.dt)
    print(info)
    env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--scenarios", nargs="*", default=SCENARIOS + ["heldout"])
    ap.add_argument("--out", default="results/demo")
    ap.add_argument("--video-scenarios", nargs="*", default=None,
                    help="scenarios to render as video (default: all)")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--scenario", default="clutter")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    if args.gui:
        gui_demo(load_policy(args.models[0]), cfg["env"], args.scenario)
        return

    os.makedirs(f"{args.out}/episodes", exist_ok=True)
    replay = []
    for path in args.models:
        run = os.path.basename(os.path.dirname(path))
        model = load_policy(path)
        env = NavEnv(**{**cfg["env"], "continuous": model.continuous})
        ex = DecisionExplainer(model, env)
        for sc in args.scenarios:
            if sc == "heldout":
                env.layout_mode = "heldout"
                ep = record_explained_episode(model, env, ex, seed=7, flags_fn=lambda r: flag_decisions(r, model, env))
                env.layout_mode = "train"
            else:
                ep = record_explained_episode(model, env, ex, seed=7, scenario=sc,
                                              flags_fn=lambda r: flag_decisions(r, model, env))
            ep["label"] = f"{run} | {sc}"
            save_episode_json(ep, f"{args.out}/episodes/{run}_{sc}.json")
            replay.append(ep)
            m = ep["metrics"]
            print(f"{run:24s} {sc:9s} success={m['success']} collision={m['collision']} steps={m['steps']}")
            if not args.no_video and (args.video_scenarios is None or sc in args.video_scenarios):
                mode = "attention" if model.has_attention() else "ig"
                files = render_episode(ep, f"{args.out}/{run}_{sc}", title=f"{run} | {sc}", ray_mode=mode)
                print("   wrote", ", ".join(files))
        env.close()
    build_replay_html(replay, f"{args.out}/replay.html")
    print("wrote", f"{args.out}/replay.html")


if __name__ == "__main__":
    main()
