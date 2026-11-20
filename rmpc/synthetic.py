"""Synthetic roads with object-level ground truth.

Each obstacle is a disk that exists with probability q, independently of the others. The
planner sees the per-cell marginal p = 1 - prod(1 - q) over the disks covering the cell.
The true chance that a path collides is 1 - prod(1 - q) over the obstacles its swept
area touches, so stage 1 can score risk exactly without sampling.
"""
from dataclasses import dataclass

import numpy as np

from .maps import Grid

LENGTH, HALF_WIDTH, RES = 60.0, 7.0, 0.25
START = np.array([2.0, 0.0, 0.0, 5.0])
GOAL = np.array([LENGTH - 5.0, 0.0])


@dataclass
class World:
    grid: Grid                 # what the planner sees
    masks: list                # one boolean cell mask per obstacle
    q: np.ndarray              # existence probability per obstacle (1 for walls)

    def true_risk(self, swept):
        hit = np.array([bool((m & swept).any()) for m in self.masks])
        return float(1 - np.prod(1 - self.q[hit]))


def make_world(rng, n_obj=(4, 8), q_range=(0.02, 0.6)):
    xs = np.arange(-5.0, LENGTH + 5.0, RES) + RES / 2
    ys = np.arange(-HALF_WIDTH - 3, HALF_WIDTH + 3, RES) + RES / 2
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    masks, q = [], []
    for _ in range(rng.integers(*n_obj, endpoint=True)):
        cx, cy = rng.uniform(10, LENGTH - 8), rng.uniform(-HALF_WIDTH + 1, HALF_WIDTH - 1)
        r = rng.uniform(0.6, 1.8)
        masks.append((X - cx) ** 2 + (Y - cy) ** 2 <= r * r)
        q.append(rng.uniform(*q_range))
    masks += [Y > HALF_WIDTH, Y < -HALF_WIDTH]
    q += [1.0, 1.0]
    q = np.array(q)
    p = np.zeros_like(X)
    for m, qi in zip(masks, q):
        p[m] = 1 - (1 - p[m]) * (1 - qi)
    return World(Grid(p, xs[0], ys[0], RES), masks, q)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def miscalibrate(p, temperature):
    """sigmoid(logit(p) / T). T < 1 sharpens probabilities, T > 1 flattens them."""
    return 1 / (1 + np.exp(-logit(p) / temperature))
