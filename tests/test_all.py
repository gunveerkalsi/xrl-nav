import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xrl_nav.agents.baselines import PotentialFieldPolicy  # noqa: E402
from xrl_nav.envs.geometry import point_clearance, shortest_path_length  # noqa: E402
from xrl_nav.envs.layouts import SCENARIOS, random_layout  # noqa: E402
from xrl_nav.envs.nav_env import NavEnv  # noqa: E402
from xrl_nav.eval.metrics import run_episode  # noqa: E402


@pytest.fixture(scope="module")
def env():
    e = NavEnv()
    yield e
    e.close()


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    from xrl_nav.agents.policies import load_policy
    from xrl_nav.agents.train import train

    out = tmp_path_factory.mktemp("run")
    train("ppo", "attention", timesteps=512, n_envs=2, out_dir=str(out), eval_every=10**9,
          hparams={"n_steps": 128, "batch_size": 64, "n_epochs": 1}, verbose=0)
    return load_policy(str(out / "model.zip"))


def test_observation_space(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)
    assert len(env.feature_names()) == obs.shape[0]
    covered = sorted(i for g in env.feature_groups().values() for i in g)
    assert covered == list(range(obs.shape[0])), "groups must partition the features"


def test_seed_determinism(env):
    o1, _ = env.reset(seed=3)
    l1 = env.layout
    o2, _ = env.reset(seed=3)
    assert np.allclose(o1, o2) and l1 == env.layout


def test_collision_and_lidar(env):
    env.reset(options={"scenario": "corridor"})
    env.pose = np.array([0.0, 0.0, np.pi / 2])  # facing the upper corridor wall at y=0.75
    env._sync_robot()
    obs = env._obs()
    front = obs[0] * env.max_range
    assert 0.5 < front < 0.6  # wall surface at 0.75 m, robot radius 0.2 m
    env.pose = np.array([0.0, 0.65, 0.0])
    env._sync_robot()
    assert env._collided()


def test_goal_reached_in_open(env):
    pf = PotentialFieldPolicy()
    row, _ = run_episode(env, pf, options={"scenario": "open"})
    assert row["success"] == 1
    assert 0.8 < row["path_efficiency"] <= 1.0


def test_layouts_are_solvable():
    rng = np.random.default_rng(0)
    for mode in ("train", "heldout"):
        for _ in range(5):
            lay = random_layout(rng, 5.0, mode)
            d = shortest_path_length(lay["obstacles"], 5.0, 0.2, lay["start"], lay["goal"])
            assert np.isfinite(d)
            assert point_clearance(*lay["start"], lay["obstacles"], 5.0) > 0.2


def test_shortest_path_straight_line():
    d = shortest_path_length([], 5.0, 0.2, [-3, 0], [3, 0])
    assert abs(d - 6.0) < 0.15


@pytest.mark.parametrize("name", SCENARIOS)
def test_scenarios_load(env, name):
    obs, _ = env.reset(options={"scenario": name})
    assert np.isfinite(obs).all()


def test_ig_completeness(tiny_model, env):
    from xrl_nav.explain.saliency import _target_value, integrated_grads

    obs, _ = env.reset(seed=1)
    X = obs[None]
    a = tiny_model.probs_np(X).argmax(1)
    base = env.baseline_obs()
    ig = integrated_grads(tiny_model, X, a, base, steps=256)
    with torch.no_grad():
        fx = _target_value(tiny_model, torch.as_tensor(X), torch.as_tensor(a)).item()
        fb = _target_value(tiny_model, torch.as_tensor(base[None]), torch.as_tensor(a)).item()
    assert abs(ig.sum() - (fx - fb)) < 0.02 * max(1.0, abs(fx - fb))


def test_shapley_efficiency(tiny_model, env):
    from xrl_nav.explain.saliency import _target_value, group_shapley

    obs, _ = env.reset(seed=2)
    X = obs[None]
    a = tiny_model.probs_np(X).argmax(1)
    base = env.baseline_obs()
    sv = group_shapley(tiny_model, X, a, env.feature_groups(), base)
    with torch.no_grad():
        fx = _target_value(tiny_model, torch.as_tensor(X), torch.as_tensor(a), prob=True).item()
        fb = _target_value(tiny_model, torch.as_tensor(base[None]), torch.as_tensor(a), prob=True).item()
    assert abs(sv.sum() - (fx - fb)) < 1e-4


def test_attention_and_explainer(tiny_model, env):
    from xrl_nav.explain.decisions import DecisionExplainer

    obs, _ = env.reset(seed=4)
    att = tiny_model.attention_np(obs)
    assert att.shape == (1, env.n_rays) and abs(att.sum() - 1) < 1e-4
    e = DecisionExplainer(tiny_model, env).explain(obs)
    assert e["text"].startswith("Chose")
    assert set(e["shapley"]) == set(env.feature_groups())


def test_deletion_curve(tiny_model, env):
    from xrl_nav.explain.faithfulness import deletion_curve
    from xrl_nav.explain.saliency import attribute

    X = np.stack([env.reset(seed=s)[0] for s in range(10)])
    a = tiny_model.probs_np(X).argmax(1)
    attr = attribute(tiny_model, X, a, "occlusion", env.baseline_obs())
    c = deletion_curve(tiny_model, X, a, attr, env.baseline_obs(), k_max=5)
    assert len(c) == 6 and c.prob_drop.iloc[0] == 0
