"""Non-learned reference controllers, used as baselines in the evaluation table."""
from __future__ import annotations

import math

import numpy as np

from ..envs.nav_env import DISCRETE_ACTIONS


class RandomPolicy:
    name = "random"

    def __init__(self, action_space, seed: int = 0):
        self.space = action_space
        self.space.seed(seed)

    def act(self, obs, deterministic=True):
        return self.space.sample()


class PotentialFieldPolicy:
    """Classic artificial potential field: attraction to the goal, repulsion from nearby lidar returns.

    Uses only the observation vector, so it sees exactly what the RL agent sees.
    """

    name = "potential_field"

    def __init__(self, n_rays: int = 24, max_range: float = 4.0, continuous: bool = False,
                 k_rep: float = 0.6, influence: float = 1.2):
        self.n_rays, self.max_range, self.continuous = n_rays, max_range, continuous
        self.k_rep, self.influence = k_rep, influence
        self.angles = np.arange(n_rays) * 2 * np.pi / n_rays

    def act(self, obs, deterministic=True):
        obs = np.asarray(obs).reshape(-1)
        d = obs[: self.n_rays] * self.max_range
        gs, gc = obs[self.n_rays + 1], obs[self.n_rays + 2]
        force = np.array([gc, gs])  # unit attraction in robot frame
        near = d < self.influence
        if near.any():
            mag = self.k_rep * (1 / np.maximum(d[near], 0.05) - 1 / self.influence)
            force -= (mag[:, None] * np.c_[np.cos(self.angles[near]), np.sin(self.angles[near])]).sum(0)
        heading = math.atan2(force[1], force[0])
        front_clear = d[[0, 1, -1]].min()
        v = 0.5 * max(math.cos(heading), 0.0) * min(1.0, front_clear / 0.8)
        w = float(np.clip(2.0 * heading, -2.0, 2.0))
        if self.continuous:
            return np.array([v / 0.5 * 2 - 1, w / 2.0], dtype=np.float32)
        cmds = np.array(DISCRETE_ACTIONS)
        return int(np.argmin((cmds[:, 0] - v) ** 2 / 0.25 + (cmds[:, 1] - w) ** 2 / 4.0))
