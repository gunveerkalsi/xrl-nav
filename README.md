# Explainable RL for Autonomous Navigation

Team GHAR (MSRIT): Hitansh Jain, Rishi Dargad, Gunveer Kalsi, Adityavikram

A reinforcement learning agent learns to drive a differential-drive robot to a goal while avoiding
obstacles in a PyBullet world exposed through the Gymnasium API. An interpretability layer then explains
each decision with attention weights, gradient saliency, Integrated Gradients, occlusion and exact
group Shapley values, checks that those explanations are faithful, and renders them alongside the
robot so that decisions can be reviewed step by step.

## Deliverables map

| Proposal deliverable | Where it lives |
|---|---|
| Simulated navigation environment (Gym + PyBullet) | `xrl_nav/envs/` (`NavEnv`, registered as `XRLNav-v0`) |
| Trained RL agent with logs and performance results | `scripts/train.py`, `runs/*/` (model, `episodes.csv`, `evaluations.csv`, SB3 logs), `results/evaluation/` |
| Interpretability module (attention / saliency + feature-level analysis) | `xrl_nav/explain/`, `scripts/explain.py`, `results/explain_*/` |
| Visual demonstration with decision explanations | `scripts/demo.py`, `results/demo/*.gif|mp4`, `results/demo/replay.html` |
| Project report | `docs/REPORT.md` (+ `docs/REPORT.pdf`) |
| Documented source repository | this repo, `tests/` |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                      # 15 tests, ~10 s
```

Python 3.10 to 3.13. CPU only; no GPU needed. On Python 3.13 `pybullet` may compile from source (5 to 10 min).

## Run everything

```bash
bash scripts/run_all.sh          # full pipeline, about 1 h on a 2-core laptop
QUICK=1 bash scripts/run_all.sh  # smoke test, a few minutes
```

Or step by step:

```bash
python scripts/train.py --algo ppo --policy mlp --seed 0            # PPO, MLP policy
python scripts/train.py --algo ppo --policy attention               # PPO, attention policy (interpretable by design)
python scripts/train.py --algo dqn --timesteps 300000               # DQN
python scripts/train.py --algo ppo --continuous                     # PPO, continuous (v, w) control
python scripts/train.py --experiment ppo_mlp                        # all seeds of an experiment in configs/default.yaml

python scripts/evaluate.py                                          # all runs + baselines, all splits
python scripts/explain.py --model runs/ppo_attention_s0/model.zip   # full interpretability analysis
python scripts/demo.py --models runs/ppo_mlp_s0/model.zip runs/ppo_attention_s0/model.zip
python scripts/demo.py --gui --models runs/ppo_mlp_s0/model.zip --scenario u_trap   # live PyBullet 3D window
python scripts/make_report_figs.py                                  # rebuild docs/ figures + REPORT.pdf
```

## Environment (`xrl_nav/envs`)

* **World**: 10 m x 10 m walled arena in PyBullet with box and cylinder obstacles. PyBullet does collision
  detection (`getClosestPoints`) and the simulated lidar (`rayTestBatch`). The robot is a 0.2 m radius
  unicycle whose pose is integrated each 0.2 s control step.
* **Observation (29)**: 24 lidar beams (every 15 deg, 4 m range, normalised), goal distance, sin/cos of the
  heading error to the goal, previous linear/angular command.
* **Actions**: discrete `forward, fwd-left, fwd-right, spin-left, spin-right` (PPO and DQN) or continuous
  `(v, w)` (PPO).
* **Reward**: progress toward goal, small time penalty, proximity penalty under 0.3 m clearance, +10 goal,
  -10 collision (episode ends).
* **Layouts**: `train` (random, 5 to 9 obstacles), `heldout` (denser, smaller obstacles plus wall segments,
  never seen in training), and hand-built scenarios `open, corridor, u_trap, clutter, gap`. Every random
  layout is checked to be solvable.

## Evaluation (`xrl_nav/eval`)

Success rate, collision rate, timeout rate, mean episode reward, **path efficiency** (geodesic optimal
path on an inflated occupancy grid divided by the distance actually driven, successful episodes only) and
minimum clearance, each with 95% confidence intervals, reported on training layouts, held-out layouts and
each scenario. Baselines: uniform random and an artificial potential field controller that sees the same
observation.

## Interpretability (`xrl_nav/explain`)

| Component | What it answers |
|---|---|
| Attention policy (`RayAttentionExtractor`) | which lidar beams the agent attends to, by construction |
| Vanilla gradient, Gradient x Input, SmoothGrad | local sensitivity of the chosen action to each input |
| Integrated Gradients | attribution that sums exactly to the change in log pi(a\|s) from a free-space baseline |
| Occlusion | probability drop when one input is reset to the baseline |
| Exact group Shapley (2^11 coalitions) | fair contribution of each semantic group: 8 lidar sectors, goal distance, goal heading, motion |
| Global importance, per-action heatmap, permutation importance | which inputs matter overall and for each action |
| Action-conditioned statistics, partial dependence | what the world looks like when each action is chosen; how the policy responds to heading and front distance |
| Surrogate decision tree | human-readable rules distilled from the policy, with fidelity |
| Deletion test vs random | faithfulness: removing top-ranked features must hurt the decision more than random ones |
| Model randomisation check | sanity: explanations must change when the weights are scrambled |
| Method agreement | rank correlation between methods, including attention vs gradients |
| Decision mining + text | automatically flags near misses, uncertain choices, turns away from a clear goal, oscillation and pre-failure steps, and writes a sentence explaining each |

## Repository layout

```
xrl_nav/
  envs/        nav_env.py (Gymnasium + PyBullet), layouts.py, geometry.py
  agents/      train.py (PPO / DQN), policies.py (attention extractor, PolicyModel wrapper), baselines.py
  eval/        metrics.py
  explain/     saliency.py, feature_analysis.py, faithfulness.py, decisions.py
  viz/         render.py (annotated frames, GIF/MP4), plots.py, replay_html.py (interactive viewer)
scripts/       train.py, evaluate.py, explain.py, demo.py, make_report_figs.py, run_all.sh
configs/       default.yaml
tests/         test_all.py
docs/          REPORT.md, REPORT.pdf, figures/
runs/          trained models and logs
results/       evaluation/, explain_<run>/, demo/
```

## Design notes and limitations

* Kinematic motion with PyBullet collision and ray casting is the usual setup for navigation policy
  learning; wheel dynamics, slip and sensor noise are out of scope.
* Attributions depend on the baseline. The default is free space (all beams at max range, no goal or
  motion information); `NavEnv.baseline_obs` can be changed.
* Attention weights show where the model routes information, not necessarily causal importance; the
  deletion test reports how faithful attention is compared with gradient methods.
