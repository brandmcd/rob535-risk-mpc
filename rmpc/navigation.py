"""Dijkstra cost-to-go on the grid, used as the MPC tracking cost and warm start.

A plain goal-distance cost made IPOPT stop in front of obstacles.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

BIG = 1e4


def cost_to_go(grid, cell_cost, goal):
    """Shortest-path cost from every cell to the goal over an 8-connected grid.

    cell_cost: per-meter cost of each cell (np.inf = blocked). Returns (V, pred) where V is
    in meters-equivalent and pred is the next cell index toward the goal.
    """
    nx, ny = cell_cost.shape
    idx = np.arange(nx * ny).reshape(nx, ny)
    rows, cols, w = [], [], []
    for di, dj in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        a = idx[max(0, -di):nx - max(0, di), max(0, -dj):ny - max(0, dj)]
        b = idx[max(0, di):nx - max(0, -di) or None, max(0, dj):ny - max(0, -dj) or None]
        a, b = a.ravel(), b.ravel()
        c = 0.5 * (cell_cost.ravel()[a] + cell_cost.ravel()[b]) * grid.res * np.hypot(di, dj)
        ok = np.isfinite(c)
        rows.append(a[ok]); cols.append(b[ok]); w.append(c[ok])
    G = coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(nx * ny, nx * ny)).tocsr()
    gi = int(np.clip(np.round((goal[0] - grid.x0) / grid.res), 0, nx - 1))
    gj = int(np.clip(np.round((goal[1] - grid.y0) / grid.res), 0, ny - 1))
    dist, pred = dijkstra(G, directed=False, indices=gi * ny + gj, return_predecessors=True)
    V = dist.reshape(nx, ny)
    V[~np.isfinite(V)] = BIG
    return V, pred


def route(grid, pred, start_xy, goal):
    """Cell-center polyline from start to goal along the Dijkstra tree."""
    nx, ny = grid.p.shape
    i = int(np.clip(np.round((start_xy[0] - grid.x0) / grid.res), 0, nx - 1))
    j = int(np.clip(np.round((start_xy[1] - grid.y0) / grid.res), 0, ny - 1))
    node, pts = i * ny + j, []
    for _ in range(nx * ny):
        pts.append((grid.xs[node // ny], grid.ys[node % ny]))
        node = pred[node]
        if node < 0:
            break
    pts = np.array(pts)
    return pts if len(pts) > 1 else np.array([start_xy, goal], float)


def guess_along(route_pts, x0, N, dt, v_guess, rear_to_center):
    """Initial guess for X: positions spaced v*dt along the route, offset to the rear axle."""
    seg = np.diff(route_pts, axis=0)
    s = np.concatenate([[0], np.cumsum(np.hypot(*seg.T))])
    v = max(v_guess, 2.0)
    targets = np.minimum(np.arange(N + 1) * v * dt, s[-1])
    cx = np.interp(targets, s, route_pts[:, 0])
    cy = np.interp(targets, s, route_pts[:, 1])
    h = np.unwrap(np.arctan2(np.gradient(cy), np.gradient(cx) + 1e-9))
    if s[-1] < 1e-6:
        h[:] = x0[2]
    h += 2 * np.pi * np.round((x0[2] - h[0]) / (2 * np.pi))
    X = np.vstack([cx - rear_to_center * np.cos(h), cy - rear_to_center * np.sin(h), h,
                   np.full(N + 1, v)])
    X[:, 0] = x0
    return X
