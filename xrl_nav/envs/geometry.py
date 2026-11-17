"""2D geometry helpers shared by the environment, metrics and visualisation.

Obstacles are described as plain dicts so they can be serialised to JSON:
    {"type": "box", "pos": [x, y], "half": [hx, hy], "yaw": rad}
    {"type": "cylinder", "pos": [x, y], "radius": r}
"""
from __future__ import annotations

import numpy as np


def point_clearance(x: float, y: float, obstacles: list[dict], arena_half: float) -> float:
    """Signed-ish clearance from point (x, y) to the nearest obstacle or wall (metres)."""
    best = min(arena_half - abs(x), arena_half - abs(y))
    for ob in obstacles:
        ox, oy = ob["pos"]
        if ob["type"] == "cylinder":
            d = np.hypot(x - ox, y - oy) - ob["radius"]
        else:
            c, s = np.cos(-ob["yaw"]), np.sin(-ob["yaw"])
            lx = c * (x - ox) - s * (y - oy)
            ly = s * (x - ox) + c * (y - oy)
            hx, hy = ob["half"]
            dx = max(abs(lx) - hx, 0.0)
            dy = max(abs(ly) - hy, 0.0)
            if abs(lx) <= hx and abs(ly) <= hy:
                d = -min(hx - abs(lx), hy - abs(ly))
            else:
                d = np.hypot(dx, dy)
        best = min(best, d)
    return float(best)


def occupancy_grid(obstacles: list[dict], arena_half: float, inflate: float, res: float = 0.1):
    """Boolean grid (True = blocked) covering [-arena_half, arena_half]^2, obstacles inflated by `inflate`."""
    n = int(round(2 * arena_half / res))
    xs = -arena_half + (np.arange(n) + 0.5) * res
    X, Y = np.meshgrid(xs, xs, indexing="xy")  # rows = y, cols = x
    blocked = (np.abs(X) > arena_half - inflate) | (np.abs(Y) > arena_half - inflate)
    for ob in obstacles:
        ox, oy = ob["pos"]
        if ob["type"] == "cylinder":
            blocked |= np.hypot(X - ox, Y - oy) < ob["radius"] + inflate
        else:
            c, s = np.cos(-ob["yaw"]), np.sin(-ob["yaw"])
            lx = c * (X - ox) - s * (Y - oy)
            ly = s * (X - ox) + c * (Y - oy)
            hx, hy = ob["half"]
            dx = np.maximum(np.abs(lx) - hx, 0.0)
            dy = np.maximum(np.abs(ly) - hy, 0.0)
            blocked |= np.hypot(dx, dy) < inflate
    return blocked, xs


def world_to_cell(x: float, y: float, arena_half: float, res: float, n: int):
    col = int(np.clip((x + arena_half) / res, 0, n - 1))
    row = int(np.clip((y + arena_half) / res, 0, n - 1))
    return row, col


def same_component(blocked: np.ndarray, a_rc, b_rc) -> bool:
    from scipy import ndimage

    lab, _ = ndimage.label(~blocked)
    return lab[a_rc] != 0 and lab[a_rc] == lab[b_rc]


def shortest_path_length(obstacles, arena_half, inflate, start, goal, res: float = 0.05) -> float:
    """Geodesic shortest path length (m) on an 8-connected occupancy grid. inf if unreachable."""
    from skimage.graph import MCP_Geometric

    blocked, _ = occupancy_grid(obstacles, arena_half, inflate, res)
    n = blocked.shape[0]
    s = world_to_cell(*start, arena_half, res, n)
    g = world_to_cell(*goal, arena_half, res, n)
    blocked[s] = False
    blocked[g] = False
    cost = np.where(blocked, np.inf, 1.0)
    mcp = MCP_Geometric(cost, fully_connected=True)
    cum, _ = mcp.find_costs([s], [g])
    d = cum[g]
    return float(d * res) if np.isfinite(d) else float("inf")
