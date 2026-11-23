"""Stage 1: baseline vs risk-budgeted MPC on synthetic maps with known risk.

Writes results/synthetic.csv. Miscalibration runs distort p with a temperature, then
undo it with one fitted on separate maps.

    python experiments/run_synthetic.py --maps 200 --workers 3
"""
import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rmpc.calib import apply_temperature, fit_temperature  # noqa: E402
from rmpc.maps import Car, footprint_cells  # noqa: E402
from rmpc.sim import run_episode  # noqa: E402
from rmpc.synthetic import GOAL, START, make_world  # noqa: E402

DELTAS = [0.01, 0.02, 0.05, 0.1, 0.2]
TAUS = [0.05, 0.1, 0.2, 0.3, 0.5]
MISCAL_T = [0.5, 2.0]
MISCAL_DELTA = 0.05
CALIB_SEED0 = 10_000   # calibration maps never overlap test maps
FIELDS = ["seed", "mode", "param", "map_T", "correction_T", "reached", "time", "path_len",
          "pred_risk", "true_risk", "fails", "over", "solve_time"]


def fit_correction(T_true, n_maps=50, samples=20, rng_seed=0):
    """Fit the temperature that undoes map_T, from sampled object outcomes on held-out maps."""
    rng = np.random.default_rng(rng_seed)
    ps, ys = [], []
    for s in range(n_maps):
        w = make_world(np.random.default_rng(CALIB_SEED0 + s))
        p_seen = apply_temperature(w.grid.p, T_true)
        # skip cells with p = 0 exactly, the logit clip would dominate the fit
        road = (np.abs(w.grid.ys)[None, :] < 7.0) & (w.grid.p > 0)
        for _ in range(samples):
            exists = rng.random(len(w.q)) < w.q
            occ = np.zeros(w.grid.p.shape, bool)
            for m, e in zip(w.masks, exists):
                if e:
                    occ |= m
            ps.append(p_seen[road])
            ys.append(occ[road])
    return fit_temperature(np.concatenate(ps), np.concatenate(ys))


def jobs(n_maps, corrections):
    for seed in range(n_maps):
        for d in DELTAS:
            yield seed, "risk", d, 1.0, 1.0
        for t in TAUS:
            yield seed, "threshold", t, 1.0, 1.0
        for T in MISCAL_T:
            yield seed, "risk", MISCAL_DELTA, T, 1.0
            yield seed, "risk", MISCAL_DELTA, T, corrections[T]


def run(job):
    seed, mode, param, map_T, corr_T = job
    car = Car()
    w = make_world(np.random.default_rng(seed))
    p = w.grid.p
    if map_T != 1.0:
        p = apply_temperature(p, map_T)
    if corr_T != 1.0:
        p = apply_temperature(p, corr_T)
    grid = w.grid.like(p)
    r = run_episode(grid, START, GOAL, mode, param, car=car)
    swept = footprint_cells(w.grid, r["poses"], car)
    seg = np.diff(r["poses"][:, :2], axis=0)
    return dict(seed=seed, mode=mode, param=param, map_T=map_T, correction_T=round(corr_T, 4),
                reached=int(r["reached"]), time=round(r["time"], 2),
                path_len=round(float(np.hypot(*seg.T).sum()), 2),
                pred_risk=round(r["pred_risk"], 5), true_risk=round(w.true_risk(swept), 5),
                fails=r["fails"], over=r["over"], solve_time=round(r["solve_time"], 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="results/synthetic.csv")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    corrections = {T: fit_correction(T) for T in MISCAL_T}
    print("fitted correction temperatures:", corrections, flush=True)
    todo = list(jobs(a.maps, corrections))
    t0 = time.time()
    with open(a.out, "w", newline="") as f, Pool(a.workers) as pool:
        wr = csv.DictWriter(f, FIELDS)
        wr.writeheader()
        for i, row in enumerate(pool.imap_unordered(run, todo, chunksize=4)):
            wr.writerow(row)
            if i % 100 == 0:
                f.flush()
                print(f"{i}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    print("done", a.out, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
