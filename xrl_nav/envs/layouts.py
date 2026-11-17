"""Obstacle layout generators.

* ``train``    : random boxes and cylinders, moderate density (training distribution)
* ``heldout``  : denser, different size mix, wall segments (never seen during training)
* named scenarios used for demos and qualitative analysis: ``open``, ``corridor``,
  ``u_trap``, ``clutter``, ``gap``
"""
from __future__ import annotations

import numpy as np

from .geometry import occupancy_grid, point_clearance, same_component, world_to_cell

ROBOT_RADIUS = 0.2


def _rand_obstacle(rng, arena_half, kind_probs=(0.5, 0.5), size=(0.25, 0.7)):
    pos = rng.uniform(-arena_half + 1.0, arena_half - 1.0, size=2).tolist()
    if rng.random() < kind_probs[0]:
        return {"type": "box", "pos": pos, "half": rng.uniform(*size, size=2).tolist(),
                "yaw": float(rng.uniform(0, np.pi))}
    return {"type": "cylinder", "pos": pos, "radius": float(rng.uniform(*size))}


def _wall(rng, arena_half):
    pos = rng.uniform(-arena_half + 1.5, arena_half - 1.5, size=2).tolist()
    return {"type": "box", "pos": pos, "half": [float(rng.uniform(0.9, 1.6)), 0.1],
            "yaw": float(rng.choice([0.0, np.pi / 2, np.pi / 4]))}


def _sample_start_goal(rng, obstacles, arena_half, min_dist):
    for _ in range(200):
        s = rng.uniform(-arena_half + 0.6, arena_half - 0.6, size=2)
        g = rng.uniform(-arena_half + 0.6, arena_half - 0.6, size=2)
        if np.linalg.norm(s - g) < min_dist:
            continue
        if point_clearance(*s, obstacles, arena_half) < ROBOT_RADIUS + 0.25:
            continue
        if point_clearance(*g, obstacles, arena_half) < ROBOT_RADIUS + 0.25:
            continue
        return s.tolist(), g.tolist()
    return None, None


def random_layout(rng, arena_half: float, mode: str = "train"):
    if mode == "train":
        n_obs = int(rng.integers(5, 10))
        maker = lambda: _rand_obstacle(rng, arena_half, (0.5, 0.5), (0.25, 0.7))  # noqa: E731
        n_walls, min_dist = 0, 4.0
    elif mode == "heldout":
        n_obs = int(rng.integers(9, 14))
        maker = lambda: _rand_obstacle(rng, arena_half, (0.3, 0.7), (0.15, 0.55))  # noqa: E731
        n_walls, min_dist = 2, 5.0
    else:
        raise ValueError(mode)

    for _ in range(100):
        obstacles = [maker() for _ in range(n_obs)] + [_wall(rng, arena_half) for _ in range(n_walls)]
        start, goal = _sample_start_goal(rng, obstacles, arena_half, min_dist)
        if start is None:
            continue
        blocked, _ = occupancy_grid(obstacles, arena_half, ROBOT_RADIUS + 0.05, res=0.1)
        n = blocked.shape[0]
        a = world_to_cell(*start, arena_half, 0.1, n)
        b = world_to_cell(*goal, arena_half, 0.1, n)
        if same_component(blocked, a, b):
            yaw = float(rng.uniform(-np.pi, np.pi))
            return {"name": mode, "obstacles": obstacles, "start": start, "goal": goal, "start_yaw": yaw}
    raise RuntimeError("could not generate a solvable layout")


def scenario(name: str, rng=None):
    """Hand-designed scenarios (arena_half = 5). Deterministic apart from optional start yaw jitter."""
    B = lambda x, y, hx, hy, yaw=0.0: {"type": "box", "pos": [x, y], "half": [hx, hy], "yaw": yaw}  # noqa: E731
    C = lambda x, y, r: {"type": "cylinder", "pos": [x, y], "radius": r}  # noqa: E731
    if name == "open":
        obs, s, g, yaw = [], [-3.5, -3.5], [3.5, 3.5], 0.0
    elif name == "corridor":
        obs = [B(0, 0.9, 3.0, 0.15), B(0, -0.9, 3.0, 0.15), C(-4, 2.5, 0.5), C(4, -2.5, 0.5)]
        s, g, yaw = [-4.2, 0.0], [4.2, 0.0], 0.0
    elif name == "u_trap":
        # A U-shaped cup opening away from the goal: greedy goal-seeking gets trapped.
        obs = [B(1.0, 0.0, 0.15, 1.6), B(0.0, 1.45, 1.0, 0.15), B(0.0, -1.45, 1.0, 0.15)]
        s, g, yaw = [-0.3, 0.0], [4.0, 0.0], 0.0
    elif name == "clutter":
        obs = [C(-2, -1, 0.4), C(-1, 1, 0.35), C(0, -0.5, 0.45), C(1, 1.5, 0.3), C(1.5, -1.5, 0.4),
               B(2.5, 0.3, 0.3, 0.5, 0.6), C(-0.5, -2.5, 0.4), B(-2.5, 2.0, 0.5, 0.3, 0.3), C(3, 2.5, 0.35)]
        s, g, yaw = [-4.0, -0.5], [4.0, 0.5], 0.3
    elif name == "gap":
        obs = [B(0, 2.9, 0.15, 2.1), B(0, -2.9, 0.15, 2.1)]
        s, g, yaw = [-3.5, -2.5], [3.5, 2.5], 0.0
    else:
        raise ValueError(f"unknown scenario {name}")
    if rng is not None:
        yaw += float(rng.uniform(-0.3, 0.3))
    return {"name": name, "obstacles": obs, "start": s, "goal": g, "start_yaw": yaw}


SCENARIOS = ["open", "corridor", "u_trap", "clutter", "gap"]
