"""Occupancy grids, hazard fields, and footprint risk/collision checks.

Arrays are indexed [ix, iy]. Cell centers are xs[ix], ys[iy].
"""
from dataclasses import dataclass

from scipy.interpolate import RegularGridInterpolator

import numpy as np
from scipy.ndimage import convolve, distance_transform_edt, gaussian_filter, maximum_filter

P_MAX = 0.999  # cap so -log(1-p) stays finite


@dataclass
class Car:
    length: float = 4.6
    width: float = 1.9
    wheelbase: float = 2.7
    rear_to_center: float = 1.35  # state is the rear axle; box center is this far ahead

    @property
    def rear_to_front(self):
        return self.rear_to_center + self.length / 2


@dataclass
class Grid:
    p: np.ndarray        # occupancy probability, shape (nx, ny)
    x0: float            # center of cell [0, 0]
    y0: float
    res: float

    @property
    def xs(self):
        return self.x0 + self.res * np.arange(self.p.shape[0])

    @property
    def ys(self):
        return self.y0 + self.res * np.arange(self.p.shape[1])

    def like(self, p):
        return Grid(p, self.x0, self.y0, self.res)


def disk_kernel(radius, res):
    r = int(np.ceil(radius / res))
    ii, jj = np.mgrid[-r:r + 1, -r:r + 1]
    k = ((ii * res) ** 2 + (jj * res) ** 2 <= radius ** 2).astype(float)
    return k / k.sum()


def hazard_field(grid, car, corr_len=4.0, smooth=0.5, margin=0.3, kind="max", corr_area=2.0):
    """Hazard per meter traveled by the car's front edge, in 1/m.

    kind="max": cells closer together than corr_len are one event (one object), events
    farther apart are independent. M(x) is the largest -log(1 - p) within half a car width
    plus `margin` of x (the margin covers the corners in turns); driving ds adds
    M * ds / corr_len, so a path through an object of probability q adds about -log(1 - q).

    kind="mean": collisions are a Poisson field with intensity -log(1 - p) / corr_area per
    square meter; the strip swept per meter has the car's width, so G is the width times
    the disk-averaged intensity. Isolated noisy cells count in proportion to their area.

    Either way P(collision) ~= 1 - exp(-sum G ds). A Gaussian blur of `smooth` meters
    gives the solver gradients.
    """
    h = -np.log1p(-np.clip(grid.p, 0.0, P_MAX))
    r = car.width / 2 + margin
    if kind == "max":
        m = maximum_filter(h, footprint=disk_kernel(r, grid.res) > 0, mode="nearest") / corr_len
    else:
        m = 2 * r * convolve(h, disk_kernel(r, grid.res), mode="nearest") / corr_area
    if smooth > 0:
        m = gaussian_filter(m, smooth / grid.res, mode="nearest")
    return m


def clearance(grid, tau):
    """Distance in meters from each cell center to the nearest cell with p > tau."""
    occ = grid.p > tau
    if not occ.any():
        return np.full(grid.p.shape, 1e3)
    return np.maximum(distance_transform_edt(~occ) * grid.res - grid.res / 2, 0.0)


def footprint_cells(grid, poses, car, step=0.25):
    """Boolean mask of cells whose centers fall inside the car box along a path.

    poses: (T, 3) rear-axle x, y, heading. The path is densified so that consecutive
    boxes are at most `step` meters apart.
    """
    poses = np.asarray(poses, float)
    dense = [poses[0]]
    for a, b in zip(poses[:-1], poses[1:]):
        n = max(1, int(np.ceil(np.hypot(*(b[:2] - a[:2])) / step)))
        dh = np.arctan2(np.sin(b[2] - a[2]), np.cos(b[2] - a[2]))
        for t in np.arange(1, n + 1) / n:
            dense.append([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * dh])
    mask = np.zeros(grid.p.shape, bool)
    xs, ys = grid.xs, grid.ys
    hl, hw = car.length / 2, car.width / 2
    reach = np.hypot(hl, hw)
    for x, y, h in dense:
        cx, cy = x + car.rear_to_center * np.cos(h), y + car.rear_to_center * np.sin(h)
        i0, i1 = np.searchsorted(xs, [cx - reach, cx + reach])
        j0, j1 = np.searchsorted(ys, [cy - reach, cy + reach])
        if i1 <= i0 or j1 <= j0:
            continue
        X, Y = np.meshgrid(xs[i0:i1] - cx, ys[j0:j1] - cy, indexing="ij")
        u = X * np.cos(h) + Y * np.sin(h)
        v = -X * np.sin(h) + Y * np.cos(h)
        mask[i0:i1, j0:j1] |= (np.abs(u) <= hl) & (np.abs(v) <= hw)
    return mask


def path_risk(p, mask):
    """1 - prod(1 - p) over the masked cells (cell-independence model)."""
    return float(-np.expm1(np.log1p(-np.clip(p[mask], 0.0, P_MAX)).sum()))


def field_lookup(grid, field):
    """Linear lookup of a grid field; points outside the grid take the nearest edge value
    (linear extrapolation can go far negative)."""
    f = RegularGridInterpolator((grid.xs, grid.ys), field)
    lo = np.array([grid.xs[0], grid.ys[0]])
    hi = np.array([grid.xs[-1], grid.ys[-1]])
    return lambda pts: f(np.clip(np.atleast_2d(pts), lo, hi))
