"""Stage 2 preprocessing: one record per nuScenes-mini keyframe.

Saves data/stage2/samples.npz with LiDAR BEV features, Occ3D obstacle/observed maps,
and a planning scenario (start speed, goal = recorded ego position `horizon` s later,
and whether the recorded path is free in the Occ3D labels).

    python experiments/prep_nusc.py
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rmpc.maps import footprint_cells  # noqa: E402
from rmpc.nusc import ZOE, NuScenes, grid_of  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/nuscenes")
    ap.add_argument("--occ", default="data/occ3d/gts")
    ap.add_argument("--out", default="data/stage2/samples.npz")
    ap.add_argument("--horizon", type=float, default=4.0)
    ap.add_argument("--min_dist", type=float, default=8.0)
    a = ap.parse_args()
    ns = NuScenes(a.root, a.occ)
    rec = {k: [] for k in ["token", "scene", "feat", "occ", "seen", "speed", "goal", "path_free",
                           "dist", "track"]}
    for si, sc in enumerate(ns.scenes):
        for tok in ns.scene_samples(sc):
            occ, seen = ns.occ_labels(tok)
            poses, ts, speed = ns.ego_track(tok, a.horizon)
            full = ts[-1] >= a.horizon - 0.1
            dist = float(np.hypot(*poses[-1, :2]))
            free = bool(full and not (footprint_cells(grid_of(occ.astype(float)), poses, ZOE) & occ).any())
            goal = poses[-1, :2] + ZOE.rear_to_center * np.array([np.cos(poses[-1, 2]), np.sin(poses[-1, 2])])
            track = np.zeros((100, 3))
            track[:len(poses)] = poses[:100]
            track[len(poses):] = poses[min(len(poses), 100) - 1]
            for k, v in [("token", tok), ("scene", si), ("feat", ns.bev_features(tok).astype(np.float16)),
                         ("occ", occ), ("seen", seen), ("speed", speed), ("goal", goal),
                         ("path_free", free and dist >= a.min_dist), ("dist", dist), ("track", track)]:
                rec[k].append(v)
        print(sc["name"], len(rec["token"]), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, **{k: np.array(v) for k, v in rec.items()})
    pf = np.array(rec["path_free"])
    print(f"saved {len(pf)} samples, {pf.sum()} usable scenarios -> {a.out}")


if __name__ == "__main__":
    main()
