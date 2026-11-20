"""Stage 2: both planners on nuScenes-mini scenarios with network occupancy probabilities.

For each usable scenario (recorded path free in Occ3D and at least 8 m long), the car
starts at the keyframe pose and speed, and the goal is where the recorded car was 4 s
later. The map is the network's out-of-fold prediction, raw or temperature-scaled with
its fold's T. A run collides if the car's swept footprint covers any observed Occ3D
obstacle cell. Writes results/nusc.csv.

    python experiments/run_nusc.py --workers 3
"""
import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rmpc.calib import sigmoid  # noqa: E402
from rmpc.maps import footprint_cells  # noqa: E402
from rmpc.nusc import ZOE, grid_of  # noqa: E402
from rmpc.sim import run_episode  # noqa: E402

DELTAS = [0.01, 0.02, 0.05, 0.1, 0.2]
TAUS = [0.1, 0.3, 0.5]
FIELDS = ["idx", "scene", "mode", "param", "calibrated", "T", "reached", "time", "path_len",
          "pred_risk", "collided", "hit_cells", "fails", "over", "solve_time"]
D = {}


def load(data, pred):
    d, p = np.load(data), np.load(pred)
    D.update(occ=d["occ"], seen=d["seen"], speed=d["speed"], goal=d["goal"], scene=d["scene"],
             logits=p["logits"], fold=p["fold"], T=p["T"], usable=d["path_free"])


def run(job):
    i, mode, param, cal = job
    T = float(D["T"][D["fold"][i]]) if cal else 1.0
    p = sigmoid(D["logits"][i].astype(np.float32) / T)
    grid = grid_of(p)
    start = np.array([0.0, 0.0, 0.0, D["speed"][i]])
    r = run_episode(grid, start, D["goal"][i], mode, param, car=ZOE, t_max=12.0)
    swept = footprint_cells(grid, r["poses"], ZOE)
    hits = int((swept & D["occ"][i]).sum())
    seg = np.diff(r["poses"][:, :2], axis=0)
    return dict(idx=i, scene=int(D["scene"][i]), mode=mode, param=param, calibrated=int(cal),
                T=round(T, 4), reached=int(r["reached"]), time=round(r["time"], 2),
                path_len=round(float(np.hypot(*seg.T).sum()), 2), pred_risk=round(r["pred_risk"], 5),
                collided=int(hits > 0), hit_cells=hits, fails=r["fails"], over=r["over"],
                solve_time=round(r["solve_time"], 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/stage2/samples.npz")
    ap.add_argument("--pred", default="data/stage2/pred.npz")
    ap.add_argument("--out", default="results/nusc.csv")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    load(a.data, a.pred)
    idx = np.where(D["usable"])[0]
    if a.limit:
        idx = idx[:a.limit]
    jobs = [(int(i), "risk", d, c) for i in idx for d in DELTAS for c in (False, True)]
    jobs += [(int(i), "threshold", t, False) for i in idx for t in TAUS]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    t0 = time.time()
    with open(a.out, "w", newline="") as f, Pool(a.workers, initializer=load, initargs=(a.data, a.pred)) as pool:
        wr = csv.DictWriter(f, FIELDS)
        wr.writeheader()
        for k, row in enumerate(pool.imap_unordered(run, jobs, chunksize=2)):
            wr.writerow(row)
            if k % 50 == 0:
                f.flush()
                print(f"{k}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    print("done", a.out, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
