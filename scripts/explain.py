#!/usr/bin/env python3
"""Run the full interpretability analysis for one trained agent.

    python scripts/explain.py --model runs/ppo_mlp_s0/model.zip
    python scripts/explain.py --model runs/ppo_attention_s0/model.zip --out results/explain_ppo_attention

Outputs (results/explain_<run>/)
    global_importance_groups.png / .csv     group importance by method (IG, gradient, SmoothGrad, occlusion,
                                            Shapley, permutation)
    feature_importance_ig.csv               per-feature mean |IG|
    shapley_per_action.png / .csv           signed group Shapley per chosen action
    action_conditioned_stats.csv            what the world looked like when each action was chosen
    pdp_heading.png, pdp_front.png          partial dependence of the policy
    surrogate_tree.txt                      distilled decision-tree rules + fidelity
    deletion_curves.png / deletion_auc.csv  faithfulness test vs random
    randomization_check.csv                 sanity check (Adebayo et al.)
    method_agreement.png / .csv             rank agreement, incl. attention when available
    attention_vs_ig.png                     (attention models)
    decisions.json / decisions.md           flagged representative & unexpected decisions with text explanations
    summary.json
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xrl_nav.agents.policies import load_policy  # noqa: E402
from xrl_nav.envs.layouts import SCENARIOS  # noqa: E402
from xrl_nav.envs.nav_env import NavEnv  # noqa: E402
from xrl_nav.eval.metrics import run_episode  # noqa: E402
from xrl_nav.explain import faithfulness as fa  # noqa: E402
from xrl_nav.explain import feature_analysis as fx  # noqa: E402
from xrl_nav.explain.decisions import DecisionExplainer, flag_decisions, select_representative  # noqa: E402
from xrl_nav.explain.saliency import attribute, group_reduce, group_shapley  # noqa: E402
from xrl_nav.viz import plots  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-episodes", type=int, default=None)
    ap.add_argument("--n-states", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run = os.path.basename(os.path.dirname(args.model))
    out = args.out or f"results/explain_{run}"
    os.makedirs(out, exist_ok=True)
    model = load_policy(args.model)
    ek = {**cfg["env"], "continuous": model.continuous}
    env = NavEnv(**ek)
    groups, fnames, anames = env.feature_groups(), env.feature_names(), env.action_names()
    gnames = list(groups)
    base = env.baseline_obs()
    n_rays, max_range = env.n_rays, env.max_range
    summary = {"model": args.model, "algo": model.algo, "continuous": model.continuous,
               "attention": model.has_attention()}

    # ---------------------------------------------------------------- dataset
    data = fx.collect_states(model, ek, args.n_episodes or cfg["explain"]["n_episodes"], scenarios=SCENARIOS)
    X, acts = data["X"], data["actions"]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=min(args.n_states or cfg["explain"]["n_states"], len(X)), replace=False)
    Xs = X[idx]
    As = acts[idx].astype(int) if not model.continuous else np.ones(len(idx), dtype=int)  # continuous: steering
    if not model.continuous:
        As = model.probs_np(Xs).argmax(1)  # explain the greedy action
    summary["n_states_total"], summary["n_states_analysed"] = int(len(X)), int(len(Xs))
    print(f"collected {len(X)} states, analysing {len(Xs)}")

    # ------------------------------------------------------ attributions
    attrs = {m: attribute(model, Xs, As, m, base) for m in
             ["gradient", "grad_x_input", "integrated_grads", "smoothgrad", "occlusion"]}
    shap = group_shapley(model, Xs, As, groups, base)
    ig = attrs["integrated_grads"]
    pd.Series(np.abs(ig).mean(0), index=fnames).sort_values(ascending=False).to_csv(f"{out}/feature_importance_ig.csv")

    # completeness check for IG (sum of attributions = f(x) - f(baseline))
    import torch

    from xrl_nav.explain.saliency import _target_value
    with torch.no_grad():
        fx_ = _target_value(model, torch.as_tensor(Xs), torch.as_tensor(As)).numpy()
        fb_ = _target_value(model, torch.as_tensor(np.broadcast_to(base, Xs.shape).copy()), torch.as_tensor(As)).numpy()
    summary["ig_completeness_mean_abs_error"] = float(np.abs(ig.sum(1) - (fx_ - fb_)).mean())

    grp = {m: pd.Series(np.abs(group_reduce(a, groups)).mean(0), index=gnames) for m, a in attrs.items()
           if m in ("gradient", "integrated_grads", "smoothgrad", "occlusion")}
    grp["group_shapley"] = pd.Series(np.abs(shap).mean(0), index=gnames)
    grp["permutation"] = fx.permutation_importance(model, Xs, groups).reindex(gnames)
    gdf = pd.DataFrame(grp)
    gdf.to_csv(f"{out}/global_importance_groups.csv")
    plots.importance_bars(grp, f"{out}/global_importance_groups.png",
                          "global importance of input groups (normalised per method)")
    summary["top_groups_shapley"] = grp["group_shapley"].sort_values(ascending=False).head(5).round(4).to_dict()

    if not model.continuous:
        sp = pd.DataFrame(shap, columns=gnames)
        sp["action"] = As
        per_action = sp.groupby("action").mean()
        per_action.to_csv(f"{out}/shapley_per_action.csv")
        plots.per_action_heatmap(per_action, anames, f"{out}/shapley_per_action.png",
                                 "mean signed Shapley contribution, by chosen action")
        stats = fx.action_conditioned_stats(X, acts, anames, n_rays, max_range)
        stats.round(3).to_csv(f"{out}/action_conditioned_stats.csv")

        # partial dependence: heading error and front distance
        grid = np.linspace(-180, 180, 37)

        def set_heading(Xc, deg):
            Xc[:, n_rays + 1] = np.sin(np.radians(deg))
            Xc[:, n_rays + 2] = np.cos(np.radians(deg))
        pdp = fx.partial_dependence(model, Xs, [], grid, transform=set_heading)
        plots.pdp_plot(grid, pdp, anames, f"{out}/pdp_heading.png", "goal heading error (deg, + = goal on the left)",
                       "partial dependence on goal direction")
        front = [n_rays - 1, 0, 1]
        dgrid = np.linspace(0.0, 1.0, 21)
        pdp2 = fx.partial_dependence(model, Xs, front, dgrid)
        plots.pdp_plot(dgrid * max_range, pdp2, anames, f"{out}/pdp_front.png", "front lidar distance (m)",
                       "partial dependence on front obstacle distance")

        tree = fx.surrogate_tree(model, X, 5, n_rays, max_range, anames)
        with open(f"{out}/surrogate_tree.txt", "w") as f:
            f.write(f"fidelity (held-out agreement with policy): {tree['fidelity']:.3f}\n")
            f.write(f"balanced fidelity (mean per-action recall): {tree['balanced_fidelity']:.3f}\n")
            f.write(f"majority-class baseline: {tree['majority_baseline']:.3f}\n\n")
            f.write(tree["rules"])
            f.write("\nfeature importance (tree):\n" + tree["feature_importance"].round(3).to_string())
        summary["surrogate_fidelity"] = tree["fidelity"]
        summary["surrogate_balanced_fidelity"] = tree["balanced_fidelity"]
        summary["surrogate_majority_baseline"] = tree["majority_baseline"]

    # ------------------------------------------------------- faithfulness
    curves = {m: fa.deletion_curve(model, Xs, As, attrs[m], base) for m in
              ["gradient", "integrated_grads", "smoothgrad", "occlusion"]}
    # Shapley expanded to features (group value split evenly over its members)
    sh_feat = np.zeros_like(Xs)
    for g, name in enumerate(gnames):
        sh_feat[:, groups[name]] = shap[:, [g]] / len(groups[name])
    curves["group_shapley"] = fa.deletion_curve(model, Xs, As, sh_feat, base)
    curves["random"] = fa.deletion_curve(model, Xs, As, None, base)
    plots.deletion_plot(curves, f"{out}/deletion_curves.png")
    auc = pd.Series({m: fa.deletion_auc(c) for m, c in curves.items()}).sort_values(ascending=False)
    auc.to_csv(f"{out}/deletion_auc.csv")
    summary["deletion_auc_k10"] = auc.round(4).to_dict()

    rc = fa.randomization_check(model, Xs[:200], As[:200], base, groups=groups)
    rc.to_csv(f"{out}/randomization_check.csv", index=False)
    summary["randomization_check"] = rc.round(3).to_dict(orient="records")

    agree = {m: attrs[m][:, :n_rays] for m in ["gradient", "integrated_grads", "smoothgrad", "occlusion"]}
    if model.has_attention():
        att = model.attention_np(Xs)
        agree["attention"] = att
        ang = np.arange(n_rays) * 360 / n_rays
        plots.attention_vs_ig(ang, att.mean(0), np.abs(ig[:, :n_rays]).mean(0), f"{out}/attention_vs_ig.png")
        att_del = np.zeros_like(Xs)
        att_del[:, :n_rays] = att
        curves_att = fa.deletion_curve(model, Xs, As, att_del, base, k_max=n_rays)
        summary["deletion_auc_k10"]["attention"] = round(fa.deletion_auc(curves_att), 4)
        pd.DataFrame({"attention": att.mean(0), "abs_ig": np.abs(ig[:, :n_rays]).mean(0)},
                     index=fnames[:n_rays]).to_csv(f"{out}/attention_profile.csv")
    M = fa.method_agreement(agree)
    M.to_csv(f"{out}/method_agreement.csv")
    plots.agreement_heatmap(M, f"{out}/method_agreement.png")

    # ------------------------------------------- representative decisions
    ex = DecisionExplainer(model, env)
    decisions = []
    for sc in SCENARIOS + ["heldout"]:
        opts = {"scenario": sc} if sc != "heldout" else {"layout_mode": "heldout"}
        row, rec = run_episode(env, model, seed=7, options=opts, record=True)
        flags = flag_decisions(rec, model, env)
        for f in select_representative(flags, 4):
            e = ex.explain(rec["obs"][f["t"]], rec["actions"][f["t"]] if not model.continuous else None)
            decisions.append({"scenario": sc, "outcome": "success" if row["success"] else
                              "collision" if row["collision"] else "timeout", **f, "text": e["text"],
                              "probs": [round(p, 3) for p in e["probs"]],
                              "shapley": {k: round(v, 4) for k, v in e["shapley"].items()}})
    json.dump(decisions, open(f"{out}/decisions.json", "w"), indent=2)
    with open(f"{out}/decisions.md", "w") as f:
        f.write("| scenario | outcome | step | flags | explanation |\n|---|---|---|---|---|\n")
        for d in decisions:
            f.write(f"| {d['scenario']} | {d['outcome']} | {d['t']} | {', '.join(d['reasons'])} | {d['text']} |\n")
    summary["n_flagged_decisions"] = len(decisions)
    json.dump(summary, open(f"{out}/summary.json", "w"), indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
