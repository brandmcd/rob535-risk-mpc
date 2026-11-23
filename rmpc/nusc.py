"""nuScenes mini + Occ3D loading: LiDAR BEV features, 2D occupancy labels, ego tracks.

Grids are the Occ3D ego frame: 200 x 200 cells of 0.4 m, x forward, y left. Obstacles
come from voxel layers 3-7 (0.2 to 2.2 m above the road).
"""
import json
import os
from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation

from .maps import Car, Grid

RES, HALF, NXY = 0.4, 40.0, 200
X0 = -HALF + RES / 2
OBSTACLE_CLASSES = list(range(0, 11)) + [15, 16]   # objects, man-made, vegetation
BAND = slice(3, 8)
ZOE = Car(length=4.08, width=1.73, wheelbase=2.59, rear_to_center=1.3)   # nuScenes car
FEAT_Z = [-1.0, 0.2, 1.0, 2.2, 5.4]                 # height bins for point counts


def grid_of(p):
    return Grid(p, X0, X0, RES)


def quat(q):
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])


def pose_matrix(rec):
    T = np.eye(4)
    T[:3, :3] = quat(rec["rotation"]).as_matrix()
    T[:3, 3] = rec["translation"]
    return T


class NuScenes:
    def __init__(self, root, occ_root, version="v1.0-mini"):
        self.root, self.occ_root = root, occ_root

        def load(t):
            with open(os.path.join(root, version, t + ".json")) as f:
                return json.load(f)
        self.scenes = load("scene")
        self.samples = {s["token"]: s for s in load("sample")}
        self.sd = {s["token"]: s for s in load("sample_data")}
        self.ego = {e["token"]: e for e in load("ego_pose")}
        self.cs = {c["token"]: c for c in load("calibrated_sensor")}
        self.scene_name = {s["token"]: s["name"] for s in self.scenes}
        self.lidar_key = {}
        for s in self.sd.values():
            if s["is_key_frame"] and "/LIDAR_TOP/" in s["filename"]:
                self.lidar_key[s["sample_token"]] = s["token"]

    def scene_samples(self, scene):
        out, tok = [], scene["first_sample_token"]
        while tok:
            out.append(tok)
            tok = self.samples[tok]["next"]
        return out

    def ego_T(self, sd_token):
        return pose_matrix(self.ego[self.sd[sd_token]["ego_pose_token"]])

    def lidar_points_ego(self, sd_token, ref_T=None):
        """LiDAR points (x, y, z, intensity) in the ego frame of ref_T (default: its own)."""
        s = self.sd[sd_token]
        pts = np.fromfile(os.path.join(self.root, s["filename"]), np.float32).reshape(-1, 5)
        T = self.ego_T(sd_token) @ pose_matrix(self.cs[s["calibrated_sensor_token"]])
        if ref_T is not None:
            T = np.linalg.inv(ref_T) @ T
        xyz = pts[:, :3] @ T[:3, :3].T + T[:3, 3]
        return np.c_[xyz, pts[:, 3]]

    def bev_features(self, sample_token, n_sweeps=1):
        """Channels: log(1 + count) per height bin, max height, mean intensity / 100,
        and a 0/1 flag for cells with any return. n_sweeps > 1 adds earlier sweeps
        moved into the keyframe's ego frame."""
        key = self.lidar_key[sample_token]
        ref = self.ego_T(key)
        pts, tok = [], key
        for _ in range(n_sweeps):
            pts.append(self.lidar_points_ego(tok, ref))
            tok = self.sd[tok]["prev"]
            if not tok:
                break
        P = np.concatenate(pts)
        i = np.floor((P[:, 0] + HALF) / RES).astype(int)
        j = np.floor((P[:, 1] + HALF) / RES).astype(int)
        ok = (i >= 0) & (i < NXY) & (j >= 0) & (j < NXY) & (P[:, 2] >= FEAT_Z[0]) & (P[:, 2] < FEAT_Z[-1])
        i, j, P = i[ok], j[ok], P[ok]
        nb = len(FEAT_Z) - 1
        f = np.zeros((nb + 3, NXY, NXY), np.float32)
        b = np.clip(np.searchsorted(FEAT_Z, P[:, 2], side="right") - 1, 0, nb - 1)
        np.add.at(f, (b, i, j), 1.0)
        f[:nb] = np.log1p(f[:nb])
        zmax = np.full((NXY, NXY), FEAT_Z[0], np.float32)
        np.maximum.at(zmax, (i, j), P[:, 2])
        f[nb] = zmax
        cnt = np.zeros((NXY, NXY), np.float32)
        np.add.at(cnt, (i, j), 1.0)
        inten = np.zeros((NXY, NXY), np.float32)
        np.add.at(inten, (i, j), P[:, 3])
        f[nb + 1] = np.where(cnt > 0, inten / np.maximum(cnt, 1) / 100.0, 0)
        f[nb + 2] = cnt > 0
        return f

    def occ_labels(self, sample_token):
        """(occupied, observed) boolean 200 x 200 maps from Occ3D."""
        scene = self.scene_name[self.samples[sample_token]["scene_token"]]
        lab = np.load(os.path.join(self.occ_root, scene, sample_token, "labels.npz"))
        sem, seen = lab["semantics"][:, :, BAND], lab["mask_lidar"][:, :, BAND] > 0
        occ = (np.isin(sem, OBSTACLE_CLASSES) & seen).any(axis=2)
        return occ, seen.any(axis=2)

    def ego_track(self, sample_token, horizon):
        """Ego poses (x, y, heading) at the LiDAR sweep rate from the keyframe to horizon
        seconds later, in the keyframe's ego frame, plus the keyframe speed (m/s)."""
        key = self.lidar_key[sample_token]
        T0inv = np.linalg.inv(self.ego_T(key))
        t0 = self.sd[key]["timestamp"]
        poses, ts, tok = [], [], key
        while tok and (self.sd[tok]["timestamp"] - t0) <= horizon * 1e6:
            T = T0inv @ self.ego_T(tok)
            poses.append([T[0, 3], T[1, 3], np.arctan2(T[1, 0], T[0, 0])])
            ts.append((self.sd[tok]["timestamp"] - t0) / 1e6)
            tok = self.sd[tok]["next"]
        poses, ts = np.array(poses), np.array(ts)
        speed = 0.0
        if len(poses) > 3:
            speed = float(np.hypot(*(poses[3, :2] - poses[0, :2])) / (ts[3] - ts[0]))
        return poses, ts, speed
