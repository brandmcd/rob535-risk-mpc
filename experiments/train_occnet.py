"""Stage 2 occupancy network: small U-Net on LiDAR BEV features.

5 folds by scene (6 train, 2 for temperature, 2 held out). Writes data/stage2/pred.npz
with logits, fold and per-fold temperature T. Any network that writes the same file
can replace it.

    python experiments/train_occnet.py --epochs 12
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rmpc.calib import fit_temperature, sigmoid  # noqa: E402

FOLDS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]   # held-out scene pairs


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, cin, w=16):
        super().__init__()
        self.e1, self.e2, self.e3 = block(cin, w), block(w, 2 * w), block(2 * w, 4 * w)
        self.d2, self.d1 = block(6 * w, 2 * w), block(3 * w, w)
        self.out = nn.Conv2d(w, 1, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(F.max_pool2d(a, 2))
        c = self.e3(F.max_pool2d(b, 2))
        b = self.d2(torch.cat([F.interpolate(c, scale_factor=2), b], 1))
        a = self.d1(torch.cat([F.interpolate(b, scale_factor=2), a], 1))
        return self.out(a)[:, 0]


def train_fold(X, Y, M, idx, epochs, seed, bs=8):
    torch.manual_seed(seed)
    net = UNet(X.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * ((len(idx) + bs - 1) // bs))
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        net.train()
        perm, tot = rng.permutation(idx), 0.0
        for k in range(0, len(perm), bs):
            b = perm[k:k + bs]
            x, y, m = (torch.from_numpy(X[b].astype(np.float32)), torch.from_numpy(Y[b].astype(np.float32)),
                       torch.from_numpy(M[b].astype(np.float32)))
            if rng.random() < 0.5:                          # mirror left-right
                x, y, m = x.flip(-1), y.flip(-1), m.flip(-1)
            loss = (F.binary_cross_entropy_with_logits(net(x), y, reduction="none") * m).sum() / m.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        print(f"  epoch {ep + 1}/{epochs} train loss {tot / len(idx):.4f}", flush=True)
    return net


@torch.no_grad()
def predict(net, X, idx, bs=8):
    net.eval()
    out = []
    for k in range(0, len(idx), bs):
        out.append(net(torch.from_numpy(X[idx[k:k + bs]].astype(np.float32))).numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/stage2/samples.npz")
    ap.add_argument("--out", default="data/stage2/pred.npz")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    d = np.load(a.data)
    X, Y, M, scene = d["feat"], d["occ"], d["seen"], d["scene"]
    logits = np.zeros(Y.shape, np.float16)
    fold_of = np.full(len(Y), -1)
    temps = []
    for f, test in enumerate(FOLDS):
        calib = FOLDS[(f + 1) % len(FOLDS)]
        tr = np.where(~np.isin(scene, test + calib))[0]
        ca = np.where(np.isin(scene, calib))[0]
        te = np.where(np.isin(scene, test))[0]
        t0 = time.time()
        print(f"fold {f}: train {len(tr)}, calib {len(ca)}, test {len(te)}", flush=True)
        net = train_fold(X, Y, M, tr, a.epochs, seed=f)
        zc = predict(net, X, ca)
        T = fit_temperature(sigmoid(zc[M[ca]]), Y[ca][M[ca]])
        temps.append(T)
        logits[te] = predict(net, X, te)
        fold_of[te] = f
        print(f"fold {f}: T = {T:.3f}, {time.time() - t0:.0f}s", flush=True)
        np.savez_compressed(a.out, logits=logits, fold=fold_of, T=np.array(temps))
    print("saved", a.out)


if __name__ == "__main__":
    main()
