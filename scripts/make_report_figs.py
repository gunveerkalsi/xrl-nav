#!/usr/bin/env python3
"""Collect result figures into docs/figures and render docs/REPORT.md to docs/REPORT.pdf.

PDF rendering uses pandoc (Markdown -> HTML) and a headless Chromium via Playwright.
Either missing -> the Markdown report is still complete; the PDF step is skipped.
"""
import glob
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "docs", "figures")

COPY = {
    "results/evaluation/training_curves.png": "training_curves.png",
    "results/evaluation/metrics.png": "metrics.png",
    "results/evaluation/trajectories_ppo_mlp.png": "trajectories_ppo_mlp.png",
    "results/evaluation/trajectories_potential_field.png": "trajectories_potential_field.png",
    "results/explain_ppo_mlp_s0/global_importance_groups.png": "importance_mlp.png",
    "results/explain_ppo_mlp_s0/shapley_per_action.png": "shapley_per_action_mlp.png",
    "results/explain_ppo_mlp_s0/pdp_heading.png": "pdp_heading_mlp.png",
    "results/explain_ppo_mlp_s0/pdp_front.png": "pdp_front_mlp.png",
    "results/explain_ppo_mlp_s0/deletion_curves.png": "deletion_mlp.png",
    "results/explain_ppo_attention_s0/attention_vs_ig.png": "attention_vs_ig.png",
    "results/explain_ppo_attention_s0/method_agreement.png": "agreement_attention.png",
    "results/explain_ppo_attention_s0/deletion_curves.png": "deletion_attention.png",
}

CSS = """body{font-family:Georgia,'Times New Roman',serif;max-width:820px;margin:28px auto;line-height:1.5;font-size:12pt;color:#111}
h1{font-size:22pt;text-align:center}h2{border-bottom:1px solid #999;margin-top:28px}h3{margin-top:20px}
img{max-width:100%;max-height:430px;display:block;margin:10px auto}table{border-collapse:collapse;margin:10px 0;font-size:9.5pt}
th,td{border:1px solid #bbb;padding:3px 6px}th{background:#eee}code{font-size:10pt}pre{background:#f5f5f5;padding:8px;font-size:9pt;overflow-x:auto}
figcaption,p.caption{text-align:center;font-size:10pt;color:#444}"""


AGENT_ORDER = ["ppo_mlp", "ppo_attention", "dqn_mlp", "ppo_mlp_cont", "potential_field", "random"]
EXPLAIN_RUNS = {"ppo_mlp": "ppo_mlp_s0", "ppo_attention": "ppo_attention_s0", "dqn_mlp": "dqn_mlp_s0",
                "ppo_mlp_cont": "ppo_mlp_cont_s0"}


def _pct(x):
    return f"{100 * x:.0f}%"


def fill_report():
    """Render docs/report_template.md into docs/REPORT.md using the numbers in results/."""
    import json

    import pandas as pd

    tpl = open(os.path.join(ROOT, "docs", "report_template.md")).read()
    summ = pd.read_csv(os.path.join(ROOT, "results/evaluation/summary.csv"))
    get = lambda a, s, k: float(summ[(summ.agent == a) & (summ.split == s)][k].iloc[0])  # noqa: E731
    seeds = lambda a: int(summ[(summ.agent == a) & (summ.split == "train")].n_seeds.iloc[0])  # noqa: E731
    agents = [a for a in AGENT_ORDER if a in set(summ.agent)]

    rows = ["| Agent | Seeds | Split | Success | Collision | Timeout | Reward | Path efficiency |",
            "|---|---|---|---|---|---|---|---|"]
    for a in agents:
        for sp in ("train", "heldout"):
            r = summ[(summ.agent == a) & (summ.split == sp)].iloc[0]
            eff = "n/a" if pd.isna(r.path_efficiency) or r.success < 0.05 else \
                f"{r.path_efficiency:.3f} ± {r.path_efficiency_ci95:.3f}"
            rows.append(f"| {a} | {r.n_seeds} | {sp} | {r.success:.2f} ± {r.success_ci95:.2f} | {r.collision:.2f} | "
                        f"{r.timeout:.2f} | {r.reward:.1f} ± {r.reward_ci95:.1f} | {eff} |")
    eval_table = "\n".join(rows) + "\n\n*Table 1. Final evaluation (100 maps per split per seed).*"

    sc = pd.read_csv(os.path.join(ROOT, "results/evaluation/scenario_success.csv"), index_col=0).reindex(agents)
    rows = ["| Agent | " + " | ".join(sc.columns) + " |", "|---|" + "---|" * len(sc.columns)]
    for a, r in sc.iterrows():
        rows.append(f"| {a} | " + " | ".join(f"{v:.2f}" for v in r.values) + " |")
    scen_table = "\n".join(rows) + "\n\n*Table 2. Success rate on hand-designed scenarios (10 runs each, start heading jittered).*"

    top_rows, del_rows, rnd_rows = [], [], []
    methods = ["integrated_grads", "occlusion", "group_shapley", "gradient", "smoothgrad", "attention", "random"]
    rmethods = ["gradient", "integrated_grads", "occlusion", "group_shapley"]
    for a, run in EXPLAIN_RUNS.items():
        p = os.path.join(ROOT, f"results/explain_{run}/summary.json")
        if not os.path.exists(p):
            continue
        s = json.load(open(p))
        top = ", ".join(f"{k.replace('lidar:', '')} ({v:.3f})" for k, v in list(s["top_groups_shapley"].items())[:4])
        top_rows.append(f"| {a} | {top} |")
        d = s["deletion_auc_k10"]
        del_rows.append(f"| {a} | " + " | ".join(f"{d[m]:.3f}" if m in d else "n/a" for m in methods) + " |")
        rc = {r["method"]: r["spearman_trained_vs_random"] for r in s["randomization_check"]}
        rnd_rows.append(f"| {a} | " + " | ".join(f"{rc[m]:.2f}" if m in rc else "n/a" for m in rmethods) + " |")
    deletion_table = ("| Agent | " + " | ".join(m.replace("_", " ") for m in methods) + " |\n|---|" + "---|" * len(methods)
                      + "\n" + "\n".join(del_rows)
                      + "\n\n*Table 3. Deletion test: mean drop in the original action's probability over the first "
                        "10 deletions (higher = more faithful). For the continuous agent the explained output is the "
                        "steering mean.*")
    rnd_table = ("| Agent | " + " | ".join(m.replace("_", " ") for m in rmethods) + " |\n|---|" + "---|" * len(rmethods)
                 + "\n" + "\n".join(rnd_rows)
                 + "\n\n*Table 4. Model randomisation check: rank correlation between attributions of the trained and "
                   "a randomly initialised network (lower is better; under 0.5 passes).*")

    mlp = json.load(open(os.path.join(ROOT, "results/explain_ppo_mlp_s0/summary.json")))
    surrogate = (f"the PPO-MLP policy's action on {_pct(mlp['surrogate_fidelity'])} of held-out states, "
                 f"against {_pct(mlp['surrogate_majority_baseline'])} for always predicting the most common "
                 f"action, with a balanced fidelity (mean per-action recall) of "
                 f"{_pct(mlp['surrogate_balanced_fidelity'])}.")

    att_h, mlp_h, pf_h = get("ppo_attention", "heldout", "success"), get("ppo_mlp", "heldout", "success"), \
        get("potential_field", "heldout", "success")
    n_att = seeds("ppo_attention")
    best_other = max(get(a, "heldout", "success") for a in agents if a not in ("ppo_attention",))
    if att_h <= best_other:
        raise SystemExit("attention is not the best held-out agent any more: rewrite the report narrative")
    att_sentence = f"{_pct(att_h)} held-out success over {n_att} seed{'s' if n_att > 1 else ''}"
    att_finding = (f"The attention policy degrades least: {_pct(att_h)} held-out success "
                   f"({n_att} seed{'s' if n_att > 1 else ''}) against {_pct(mlp_h)} for the MLP policy trained with "
                   f"the same algorithm and budget (3 seeds), {_pct(get('dqn_mlp', 'heldout', 'success'))} for DQN "
                   f"and {_pct(get('ppo_mlp_cont', 'heldout', 'success'))} for continuous PPO. Treating each "
                   f"beam as a token with its own angle appears to transfer better to new obstacle arrangements "
                   f"than a flat input vector.")
    seed_limit = ("PPO-MLP and PPO-attention use three training seeds each;" if n_att >= 3 else
                  f"PPO-MLP uses three seeds and PPO-attention {n_att};")
    out = (tpl.replace("ATT_HELDOUT_SENTENCE", att_sentence).replace("MLP_HELDOUT", _pct(mlp_h))
           .replace("PF_HELDOUT", _pct(pf_h)).replace("ATT_SEEDS", str(n_att)).replace("EVAL_TABLE", eval_table)
           .replace("SCEN_TABLE", scen_table).replace("ATT_FINDING", att_finding)
           .replace("TOP_GROUPS_TABLE", "\n".join(top_rows)).replace("SURROGATE_SENTENCE", surrogate)
           .replace("DELETION_TABLE", deletion_table).replace("RANDOMIZATION_TABLE", rnd_table)
           .replace("ATT_SEED_LIMIT", seed_limit))
    open(os.path.join(ROOT, "docs", "REPORT.md"), "w").write(out)
    print("wrote docs/REPORT.md")


def main():
    os.makedirs(FIG, exist_ok=True)
    fill_report()
    for src, dst in COPY.items():
        p = os.path.join(ROOT, src)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(FIG, dst))
    # one key frame from each demo video
    try:
        import imageio.v2 as imageio

        for mp4 in sorted(glob.glob(os.path.join(ROOT, "results/demo/*.mp4"))):
            r = imageio.get_reader(mp4)
            n = r.count_frames()
            frame = r.get_data(int(n * 0.4))
            imageio.imwrite(os.path.join(FIG, "frame_" + os.path.basename(mp4).replace(".mp4", ".png")), frame)
    except Exception as e:  # pragma: no cover
        print("frame extraction skipped:", e)

    md = os.path.join(ROOT, "docs", "REPORT.md")
    html = os.path.join(ROOT, "docs", "REPORT.html")
    pdf = os.path.join(ROOT, "docs", "REPORT.pdf")
    if not shutil.which("pandoc"):
        print("pandoc not found; skipping PDF")
        return
    css_path = os.path.join(ROOT, "docs", "report.css")
    open(css_path, "w").write(CSS)
    subprocess.run(["pandoc", md, "-f", "markdown-implicit_figures", "-s", "-o", html, "--css", "report.css",
                    "--metadata", "pagetitle=XRL Navigation Report",
                    "--resource-path", os.path.join(ROOT, "docs")], check=True, cwd=os.path.join(ROOT, "docs"))
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright not installed; HTML report written to", html)
        return
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page()
        page.goto("file://" + html)
        page.pdf(path=pdf, format="A4", margin={"top": "18mm", "bottom": "18mm", "left": "16mm", "right": "16mm"},
                 print_background=True)
        b.close()
    os.remove(html)
    print("wrote", pdf)


if __name__ == "__main__":
    sys.exit(main())
