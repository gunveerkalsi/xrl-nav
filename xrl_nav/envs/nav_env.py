"""Gymnasium navigation environment backed by PyBullet.

PyBullet provides the 3D world (floor, walls, obstacles, robot body), collision
detection (``getClosestPoints``) and the simulated lidar (``rayTestBatch``).
The robot is a differential-drive (unicycle) base; its pose is integrated
kinematically each control step and written back into PyBullet, which is the
standard setup for navigation-policy learning and keeps training fast on CPU.

Observation (all roughly in [-1, 1]) -- see ``feature_names()``:
    ray_000 ... ray_345 : normalised lidar distance per beam (0 = touching, 1 = max range),
                          beam k points at k*360/N degrees relative to the heading (CCW = left)
    goal_dist           : distance to goal / arena diagonal
    goal_sin, goal_cos  : sine / cosine of the heading error to the goal
    lin_vel, ang_vel    : previous command, normalised

Actions:
    discrete (default)  : 0 forward, 1 forward-left, 2 forward-right, 3 spin-left, 4 spin-right
    continuous          : [v, w] in [-1, 1]^2 mapped to v in [0, v_max], w in [-w_max, w_max]
"""
from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium import spaces

from .layouts import ROBOT_RADIUS, random_layout, scenario

DISCRETE_ACTIONS = [  # (v [m/s], w [rad/s])
    (0.50, 0.0),
    (0.35, 1.0),
    (0.35, -1.0),
    (0.05, 2.0),
    (0.05, -2.0),
]
ACTION_NAMES = ["forward", "fwd-left", "fwd-right", "spin-left", "spin-right"]

SECTOR_NAMES = ["front", "front-left", "left", "back-left", "back", "back-right", "right", "front-right"]


class NavEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 10}

    def __init__(
        self,
        n_rays: int = 24,
        max_range: float = 4.0,
        arena_half: float = 5.0,
        dt: float = 0.2,
        max_steps: int = 400,
        layout_mode: str = "train",
        scenario_name: str | None = None,
        continuous: bool = False,
        goal_radius: float = 0.35,
        terminate_on_collision: bool = True,
        render_mode: str | None = None,
        reward_cfg: dict | None = None,
    ):
        super().__init__()
        assert n_rays % 8 == 0, "n_rays must be a multiple of 8 (sector grouping)"
        self.n_rays, self.max_range, self.arena_half = n_rays, max_range, arena_half
        self.dt, self.max_steps = dt, max_steps
        self.layout_mode, self.scenario_name = layout_mode, scenario_name
        self.continuous, self.goal_radius = continuous, goal_radius
        self.terminate_on_collision = terminate_on_collision
        self.render_mode = render_mode
        self.v_max, self.w_max = 0.5, 2.0
        self.diag = 2 * math.sqrt(2) * arena_half
        self.rcfg = {"progress": 3.0, "step": -0.005, "goal": 10.0, "collision": -10.0,
                     "proximity": -0.1, "proximity_dist": 0.3}
        self.rcfg.update(reward_cfg or {})

        self.ray_angles = np.arange(n_rays) * 2 * np.pi / n_rays
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(n_rays + 5,), dtype=np.float32)
        if continuous:
            self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(len(DISCRETE_ACTIONS))

        self.client = p.connect(p.GUI if render_mode == "human" else p.DIRECT)
        self._obstacle_ids: list[int] = []
        self._static_ids: list[int] = []
        self.robot_id = None
        self.goal_id = None
        self.layout: dict | None = None
        self._build_static()

    # ------------------------------------------------------------------ world
    def _build_static(self):
        c = self.client
        p.resetSimulation(physicsClientId=c)
        p.setGravity(0, 0, -9.81, physicsClientId=c)
        h = self.arena_half
        floor = p.createCollisionShape(p.GEOM_BOX, halfExtents=[h + 0.2, h + 0.2, 0.05], physicsClientId=c)
        floor_v = p.createVisualShape(p.GEOM_BOX, halfExtents=[h + 0.2, h + 0.2, 0.05],
                                      rgbaColor=[0.92, 0.92, 0.9, 1], physicsClientId=c)
        p.createMultiBody(0, floor, floor_v, [0, 0, -0.05], physicsClientId=c)
        for (x, y, hx, hy) in [(0, h + 0.1, h + 0.2, 0.1), (0, -h - 0.1, h + 0.2, 0.1),
                               (h + 0.1, 0, 0.1, h + 0.2), (-h - 0.1, 0, 0.1, h + 0.2)]:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, 0.3], physicsClientId=c)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, 0.3], rgbaColor=[0.3, 0.3, 0.35, 1],
                                      physicsClientId=c)
            self._static_ids.append(p.createMultiBody(0, col, vis, [x, y, 0.3], physicsClientId=c))
        rc = p.createCollisionShape(p.GEOM_CYLINDER, radius=ROBOT_RADIUS, height=0.3, physicsClientId=c)
        rv = p.createVisualShape(p.GEOM_CYLINDER, radius=ROBOT_RADIUS, length=0.3, rgbaColor=[0.1, 0.45, 0.9, 1],
                                 physicsClientId=c)
        self.robot_id = p.createMultiBody(0, rc, rv, [0, 0, 0.16], physicsClientId=c)
        gv = p.createVisualShape(p.GEOM_SPHERE, radius=self.goal_radius * 0.6, rgbaColor=[0.1, 0.8, 0.3, 0.8],
                                 physicsClientId=c)
        self.goal_id = p.createMultiBody(0, -1, gv, [0, 0, 0.2], physicsClientId=c)

    def _spawn_obstacles(self, obstacles):
        c = self.client
        for oid in self._obstacle_ids:
            p.removeBody(oid, physicsClientId=c)
        self._obstacle_ids = []
        for ob in obstacles:
            if ob["type"] == "box":
                he = [ob["half"][0], ob["half"][1], 0.3]
                col = p.createCollisionShape(p.GEOM_BOX, halfExtents=he, physicsClientId=c)
                vis = p.createVisualShape(p.GEOM_BOX, halfExtents=he, rgbaColor=[0.75, 0.4, 0.2, 1], physicsClientId=c)
                orn = p.getQuaternionFromEuler([0, 0, ob["yaw"]])
            else:
                col = p.createCollisionShape(p.GEOM_CYLINDER, radius=ob["radius"], height=0.6, physicsClientId=c)
                vis = p.createVisualShape(p.GEOM_CYLINDER, radius=ob["radius"], length=0.6,
                                          rgbaColor=[0.8, 0.55, 0.2, 1], physicsClientId=c)
                orn = [0, 0, 0, 1]
            self._obstacle_ids.append(
                p.createMultiBody(0, col, vis, [ob["pos"][0], ob["pos"][1], 0.3], orn, physicsClientId=c))

    # --------------------------------------------------------------- sensing
    def _lidar(self) -> np.ndarray:
        x, y, th = self.pose
        ang = th + self.ray_angles
        d = np.stack([np.cos(ang), np.sin(ang)], 1)
        z = 0.16
        starts = np.c_[x + d[:, 0] * (ROBOT_RADIUS + 0.01), y + d[:, 1] * (ROBOT_RADIUS + 0.01), np.full(len(ang), z)]
        ends = np.c_[x + d[:, 0] * (ROBOT_RADIUS + self.max_range), y + d[:, 1] * (ROBOT_RADIUS + self.max_range),
                     np.full(len(ang), z)]
        res = p.rayTestBatch(starts.tolist(), ends.tolist(), physicsClientId=self.client)
        frac = np.array([r[2] if r[0] != -1 else 1.0 for r in res], dtype=np.float32)
        return frac  # fraction of max range, measured from the robot surface

    def _collided(self) -> bool:
        for oid in self._obstacle_ids + self._static_ids:
            if p.getClosestPoints(self.robot_id, oid, 0.0, physicsClientId=self.client):
                return True
        return False

    def _goal_terms(self):
        x, y, th = self.pose
        gx, gy = self.layout["goal"]
        dist = math.hypot(gx - x, gy - y)
        err = math.atan2(gy - y, gx - x) - th
        err = (err + math.pi) % (2 * math.pi) - math.pi
        return dist, err

    def _obs(self) -> np.ndarray:
        self._last_rays = self._lidar()
        dist, err = self._goal_terms()
        v, w = self._last_cmd
        tail = np.array([min(dist / self.diag, 1.0), math.sin(err), math.cos(err), v / self.v_max, w / self.w_max],
                        dtype=np.float32)
        return np.concatenate([self._last_rays, tail]).astype(np.float32)

    # ------------------------------------------------------------------- API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        if "layout" in options:
            self.layout = options["layout"]
        elif options.get("scenario") or self.scenario_name:
            self.layout = scenario(options.get("scenario") or self.scenario_name, self.np_random)
        else:
            self.layout = random_layout(self.np_random, self.arena_half, options.get("layout_mode", self.layout_mode))
        self._spawn_obstacles(self.layout["obstacles"])
        sx, sy = self.layout["start"]
        self.pose = np.array([sx, sy, self.layout["start_yaw"]], dtype=np.float64)
        self._sync_robot()
        p.resetBasePositionAndOrientation(self.goal_id, [*self.layout["goal"], 0.2], [0, 0, 0, 1],
                                          physicsClientId=self.client)
        self._last_cmd = (0.0, 0.0)
        self.steps, self.path_length = 0, 0.0
        self.trajectory = [self.pose.copy()]
        self.prev_dist, _ = self._goal_terms()
        self.min_clearance = np.inf
        obs = self._obs()
        return obs, self._info(False, False)

    def _sync_robot(self):
        x, y, th = self.pose
        p.resetBasePositionAndOrientation(self.robot_id, [x, y, 0.16], p.getQuaternionFromEuler([0, 0, th]),
                                          physicsClientId=self.client)

    def action_to_cmd(self, action) -> tuple[float, float]:
        if self.continuous:
            a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
            return float((a[0] + 1) / 2 * self.v_max), float(a[1] * self.w_max)
        return DISCRETE_ACTIONS[int(action)]

    def step(self, action):
        v, w = self.action_to_cmd(action)
        x, y, th = self.pose
        n_sub = 4  # integrate the unicycle model in sub-steps for accuracy
        for _ in range(n_sub):
            th += w * self.dt / n_sub
            x += v * math.cos(th) * self.dt / n_sub
            y += v * math.sin(th) * self.dt / n_sub
        th = (th + math.pi) % (2 * math.pi) - math.pi
        self.path_length += math.hypot(x - self.pose[0], y - self.pose[1])
        self.pose = np.array([x, y, th])
        self._sync_robot()
        self._last_cmd = (v, w)
        self.steps += 1
        self.trajectory.append(self.pose.copy())

        collided = self._collided()
        obs = self._obs()
        dist, _ = self._goal_terms()
        min_ray = float(self._last_rays.min() * self.max_range)
        self.min_clearance = min(self.min_clearance, min_ray)

        r = self.rcfg["progress"] * (self.prev_dist - dist) + self.rcfg["step"]
        if min_ray < self.rcfg["proximity_dist"]:
            r += self.rcfg["proximity"] * (1 - min_ray / self.rcfg["proximity_dist"])
        self.prev_dist = dist
        success = dist < self.goal_radius
        terminated = False
        if success:
            r += self.rcfg["goal"]
            terminated = True
        elif collided:
            r += self.rcfg["collision"]
            terminated = self.terminate_on_collision
        truncated = (not terminated) and self.steps >= self.max_steps
        return obs, float(r), terminated, truncated, self._info(success, collided)

    def _info(self, success, collided) -> dict[str, Any]:
        return {"success": bool(success), "collision": bool(collided), "steps": self.steps,
                "path_length": self.path_length, "pose": self.pose.tolist(),
                "min_clearance": float(self.min_clearance)}

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        x, y, th = self.pose
        view = p.computeViewMatrix([x - 2.5 * math.cos(th), y - 2.5 * math.sin(th), 3.0], [x, y, 0], [0, 0, 1],
                                   physicsClientId=self.client)
        proj = p.computeProjectionMatrixFOV(60, 4 / 3, 0.1, 30, physicsClientId=self.client)
        _, _, rgb, _, _ = p.getCameraImage(320, 240, view, proj, renderer=p.ER_TINY_RENDERER,
                                           physicsClientId=self.client)
        return np.asarray(rgb, dtype=np.uint8).reshape(240, 320, 4)[:, :, :3]

    def close(self):
        if p.isConnected(self.client):
            p.disconnect(self.client)

    # ----------------------------------------------------------- metadata
    def feature_names(self) -> list[str]:
        return [f"ray_{int(round(np.degrees(a))):03d}" for a in self.ray_angles] + \
            ["goal_dist", "goal_sin", "goal_cos", "lin_vel", "ang_vel"]

    def feature_groups(self) -> dict[str, list[int]]:
        """Semantic groups used for group-level attribution (8 lidar sectors + goal + motion)."""
        per = self.n_rays // 8
        groups = {}
        for k, name in enumerate(SECTOR_NAMES):
            centre = k * per
            groups[f"lidar:{name}"] = [(centre + j) % self.n_rays for j in range(-(per // 2), per - per // 2)]
        n = self.n_rays
        groups["goal_distance"] = [n]
        groups["goal_heading"] = [n + 1, n + 2]
        groups["motion"] = [n + 3, n + 4]
        return groups

    def action_names(self) -> list[str]:
        return ["v", "w"] if self.continuous else list(ACTION_NAMES)

    def baseline_obs(self, obs: np.ndarray | None = None) -> np.ndarray:
        """Reference input for attribution: free space (all rays max), no motion, goal terms kept neutral."""
        b = np.zeros(self.observation_space.shape, dtype=np.float32)
        b[: self.n_rays] = 1.0
        return b
