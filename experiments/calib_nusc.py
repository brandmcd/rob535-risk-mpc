"""Stage 2 cell-level calibration check: reliability and ECE of the network's
out-of-fold probabilities on observed cells, before and after temperature scaling.

Writes results/nusc_calibration.json and figures/nusc_reliability.png.

    python experiments/calib_nusc.py
"""
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rmpc.calib import ece, reliability, sigmoid  # noqa: E402

EDGES = np.array([0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0])


def main():
    d, p = np.load("data/stage2/samples.npz"), np.load("data/stage2/pred.npz")
    seen, occ, fold = d["seen"], d["occ"], p["fold"]
    z = p["logits"].astype(np.float32)
    T = p["T"][fold][:, None, None]
    raw = sigmoid(z)[seen]
    cal = sigmoid(z / T)[seen]
    y = occ[seen]
    nll = lambda q: float(-np.mean(y * np.log(np.clip(q, 1e-7, 1)) + (~y) * np.log(np.clip(1 - q, 1e-7, 1))))
    out = dict(cells=int(y.size), base_rate=float(y.mean()), T_per_fold=p["T"].round(4).tolist(),
               ece_raw=ece(raw, y, edges=EDGES), ece_cal=ece(cal, y, edges=EDGES),
               nll_raw=nll(raw), nll_cal=nll(cal))
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    for q, name, c in [(raw, "raw", "C3"), (cal, "temperature-scaled", "C0")]:
        conf, freq, cnt = reliability(q, y, edges=EDGES)
        out[f"bins_{name}"] = dict(conf=np.nan_to_num(conf).round(4).tolist(),
                                   freq=np.nan_to_num(freq).round(4).tolist(), count=cnt.tolist())
        ok = cnt > 100
        ax[0].plot(conf[ok], freq[ok], "o-", color=c, label=f"{name} (ECE {ece(q, y, edges=EDGES):.3f})")
        ax[1].hist(q, bins=50, range=(0, 1), histtype="step", color=c, log=True, label=name)
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set(xlabel="predicted occupancy probability", ylabel="observed occupied fraction",
              title="Reliability, observed cells, out-of-fold")
    ax[0].legend()
    ax[1].set(xlabel="predicted probability", ylabel="cells", title="Prediction histogram")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig("figures/nusc_reliability.png", dpi=150)
    with open("results/nusc_calibration.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("bins")}, indent=1))


if __name__ == "__main__":
    main()
