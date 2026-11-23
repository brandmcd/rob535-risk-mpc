"""Stage 2 path-level calibration.

Draws random 4 s rollouts per scenario, computes each path's hazard H (corr_len = 1) and
whether it hits an Occ3D obstacle, then fits a scale s in 1 - exp(-s * H) per held-out
fold. Writes results/path_calib.npz and results/path_scale.json.

    python experiments/path_calib.py --workers 3
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rmpc.calib import sigmoid  # noqa: E402
from rmpc.maps import field_lookup, footprint_cells, hazard_field  # noqa: E402
from rmpc.nusc import ZOE, grid_of  # noqa: E402
from rmpc.sim import front, step  # noqa: E402

D = {}


def load(data, pred):
    d, p = np.load(data), np.load(pred)
    D.update(occ=d["occ"], speed=d["speed"], scene=d["scene"], usable=d["path_free"],
             logits=p["logits"], fold=p["fold"], T=p["T"])


def rollouts(i, n, horizon=4.0, dt=0.1, seed=0):
    rng = np.random.default_rng(seed + i)
    out = []
    for _ in range(n):
        x = np.array([0.0, 0.0, 0.0, D["speed"][i]])
        k_switch = rng.integers(10, 30)
        smax = rng.choice([0.02, 0.05, 0.1, 0.2])     # spread from on-road to off-road paths
        u1 = [rng.uniform(-2.5, 1.5), rng.uniform(-smax, smax)]
        u2 = [rng.uniform(-2.5, 1.5), rng.uniform(-smax, smax)]
        poses = [x.copy()]
        for k in range(int(horizon / dt)):
            x = step(x.copy(), u1 if k < k_switch else u2, ZOE, dt, sub=1)
            poses.append(x.copy())
        out.append(np.array(poses))
    return out


def path_hazard(H, poses):
    tot = 0.0
    for a, b in zip(poses[:-1], poses[1:]):
        f0, f1 = front(a, ZOE), front(b, ZOE)
        tot += float(H([(f0 + f1) / 2])[0]) * np.hypot(*(f1 - f0))
    return tot


def run(args):
    i, n = args
    z = D["logits"][i].astype(np.float32)
    T = float(D["T"][D["fold"][i]])
    maps = {"raw": sigmoid(z), "cal": sigmoid(z / T)}
    grid = grid_of(maps["raw"])
    H = {}
    for name, p in maps.items():
        for kind in ("max", "mean"):
            f = hazard_field(grid.like(p), ZOE, corr_len=1.0, kind=kind, corr_area=1.0)
            H[f"{kind}_{name}"] = field_lookup(grid, f)
    rows = []
    for poses in rollouts(i, n):
        swept = footprint_cells(grid, poses[:, :3], ZOE)
        hits = int((swept & D["occ"][i]).sum())
        seg = np.diff(poses[:, :2], axis=0)
        row = dict(idx=i, scene=int(D["scene"][i]), fold=int(D["fold"][i]), hits=hits, collided=int(hits > 0),
                   length=float(np.hypot(*seg.T).sum()))
        for k, h in H.items():
            row["H_" + k] = path_hazard(h, poses)
        rows.append(row)
    return rows


def fit_scale(H, y):
    def nll(log_s):
        q = np.clip(-np.expm1(-np.exp(log_s) * H), 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(q) + (1 - y) * np.log1p(-q))
    r = minimize_scalar(nll, bounds=(-8, 4), method="bounded")
    return float(np.exp(r.x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/stage2/samples.npz")
    ap.add_argument("--pred", default="data/stage2/pred.npz")
    ap.add_argument("--per", type=int, default=60)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    load(a.data, a.pred)
    idx = np.where(D["usable"])[0]
    with Pool(a.workers, initializer=load, initargs=(a.data, a.pred)) as pool:
        rows = [r for rs in pool.imap_unordered(run, [(int(i), a.per) for i in idx]) for r in rs]
    cols = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    os.makedirs("results", exist_ok=True)
    np.savez_compressed("results/path_calib.npz", **cols)
    y = cols["collided"]
    scales = {}
    for key in [k for k in cols if k.startswith("H_")]:
        scales[key] = {}
        for f in sorted(set(cols["fold"].tolist())):
            tr = cols["fold"] != f
            scales[key][f] = fit_scale(cols[key][tr], y[tr])
        scales[key]["all"] = fit_scale(cols[key], y)
    with open("results/path_scale.json", "w") as fh:
        json.dump(scales, fh, indent=1)
    print(f"{len(y)} paths, {y.mean():.3f} collide")
    print(json.dumps(scales, indent=1))


if __name__ == "__main__":
    main()
