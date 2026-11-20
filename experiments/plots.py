"""Figures and summary tables from results/synthetic.csv and results/nusc.csv.

    python experiments/plots.py
"""
import json
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rmpc.calib import wilson  # noqa: E402

os.makedirs("figures", exist_ok=True)
summary = {}


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def synthetic():
    df = pd.read_csv("results/synthetic.csv")
    base = df[df.map_T == 1.0]
    rows = []
    for (mode, param), g in base.groupby(["mode", "param"]):
        lo, hi = boot_ci(g.true_risk)
        rows.append(dict(mode=mode, param=param, n=len(g), reach=g.reached.mean(),
                         true_risk=g.true_risk.mean(), lo=lo, hi=hi, pred_risk=g.pred_risk.mean(),
                         path_len=g.path_len.mean(), time=g[g.reached == 1].time.mean(),
                         max_over_budget=(g.pred_risk > param * 1.1).mean() if mode == "risk" else np.nan))
    t = pd.DataFrame(rows)
    summary["synthetic"] = t.round(4).to_dict("records")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for mode, c, mk in [("risk", "C0", "o"), ("threshold", "C3", "s")]:
        s = t[t["mode"] == mode]
        ax[0].errorbar(s.true_risk, s.reach, xerr=[s.true_risk - s.lo, s.hi - s.true_risk], fmt=mk + "-",
                       color=c, label="risk-budgeted MPC (delta)" if mode == "risk" else "threshold MPC (tau)")
        for _, r in s.iterrows():
            ax[0].annotate(f"{r.param:g}", (r.true_risk, r.reach), textcoords="offset points", xytext=(4, 4),
                           fontsize=8, color=c)
    ax[0].set(xlabel="mean true collision probability", ylabel="fraction of runs reaching the goal",
              title="Stage 1: safety vs progress (200 maps)")
    ax[0].legend()
    s = t[t["mode"] == "risk"]
    ax[1].errorbar(s.param, s.true_risk, yerr=[s.true_risk - s.lo, s.hi - s.true_risk], fmt="o-", label="true risk")
    ax[1].plot(s.param, s.pred_risk, "^--", label="risk predicted from the map")
    lim = [0, max(s.param) * 1.1]
    ax[1].plot(lim, lim, "k:", lw=0.8, label="budget")
    ax[1].set(xlabel="risk budget delta", ylabel="mean collision probability",
              title="Stage 1: risk vs budget, calibrated maps")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig("figures/synthetic_tradeoff.png", dpi=150)

    mis = df[(df["mode"] == "risk") & (df.param == 0.05)]
    rows = []
    for (mT, cT), g in mis.groupby(["map_T", "correction_T"]):
        lo, hi = boot_ci(g.true_risk)
        rows.append(dict(map_T=mT, correction_T=cT, n=len(g), true_risk=g.true_risk.mean(), lo=lo, hi=hi,
                         pred_risk=g.pred_risk.mean(), reach=g.reached.mean()))
    m = pd.DataFrame(rows)
    summary["synthetic_miscalibration"] = m.round(4).to_dict("records")
    fig, ax = plt.subplots(figsize=(6, 4.2))
    labels = [("T=%g" % r.map_T) + (" corrected" if r.correction_T != 1 else "") for _, r in m.iterrows()]
    x = np.arange(len(m))
    ax.bar(x - 0.2, m.true_risk, 0.4, yerr=[m.true_risk - m.lo, m.hi - m.true_risk], label="true risk")
    ax.bar(x + 0.2, m.pred_risk, 0.4, label="risk the planner believed")
    ax.axhline(0.05, color="k", ls=":", label="budget 0.05")
    ax.set_xticks(x, labels, rotation=20)
    ax.set(ylabel="mean collision probability", title="Stage 1: miscalibrated maps, delta = 0.05")
    ax.legend()
    plt.tight_layout()
    plt.savefig("figures/synthetic_miscalibration.png", dpi=150)


VARIANTS = [("raw", "C3", "raw net, l = 4 m"), ("cal", "C1", "temperature-scaled, l = 4 m"),
            ("path", "C0", "temperature + path scale")]


def nusc():
    if not os.path.exists("results/nusc.csv"):
        return
    df = pd.read_csv("results/nusc.csv")
    rows = []
    for (mode, param, var), g in df.groupby(["mode", "param", "variant"]):
        k, n = int(g.collided.sum()), len(g)
        lo, hi = wilson(k, n)
        rows.append(dict(mode=mode, param=param, variant=var, n=n, collisions=k, rate=k / n, lo=lo, hi=hi,
                         pred_risk=g.pred_risk.mean(), reach=g.reached.mean(), corr_len=g.corr_len.mean(),
                         time=g[g.reached == 1].time.mean(), path_len=g.path_len.mean()))
    t = pd.DataFrame(rows)
    summary["nusc"] = t.round(4).to_dict("records")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for var, c, name in VARIANTS:
        s = t[(t["mode"] == "risk") & (t.variant == var)]
        if s.empty:
            continue
        ax[0].errorbar(s.param, s.rate, yerr=[s.rate - s.lo, s.hi - s.rate], fmt="o-", color=c,
                       label=f"observed, {name}")
        ax[0].plot(s.param, s.pred_risk, "^--", color=c, alpha=0.6, label=f"predicted, {name}")
        ax[1].plot(s.rate, s.reach, "o-", color=c, label=f"risk MPC, {name}")
    lim = [0, t[t["mode"] == "risk"].param.max() * 1.1]
    ax[0].plot(lim, lim, "k:", lw=0.8, label="budget")
    ax[0].set(xlabel="risk budget delta", ylabel="collision rate (Occ3D labels)",
              title=f"Stage 2: nuScenes mini, {int(t.n.max())} scenarios")
    ax[0].legend(fontsize=7)
    s = t[t["mode"] == "threshold"]
    ax[1].plot(s.rate, s.reach, "s-", color="C2", label="threshold MPC (tau)")
    for _, r in s.iterrows():
        ax[1].annotate(f"{r.param:g}", (r.rate, r.reach), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax[1].set(xlabel="collision rate", ylabel="fraction reaching the goal", title="Stage 2: safety vs progress")
    ax[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("figures/nusc_planning.png", dpi=150)


PATH_EDGES = np.array([0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0])


def path_level():
    if not os.path.exists("results/path_calib.npz"):
        return
    d = np.load("results/path_calib.npz")
    with open("results/path_scale.json") as f:
        sc = json.load(f)
    y, fold = d["collided"], d["fold"]
    s_fold = np.array([sc["H_max_cal"][str(f)] for f in fold])
    preds = {"raw net, l = 4 m": -np.expm1(-d["H_max_raw"] / 4),
             "temperature-scaled, l = 4 m": -np.expm1(-d["H_max_cal"] / 4),
             "temperature + path scale (other folds)": -np.expm1(-s_fold * d["H_max_cal"])}
    out = dict(paths=int(len(y)), collide=float(y.mean()), scale=sc)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for (name, q), c in zip(preds.items(), ["C3", "C1", "C0"]):
        idx = np.clip(np.digitize(q, PATH_EDGES[1:-1]), 0, len(PATH_EDGES) - 2)
        xs, ys, lo, hi, ece = [], [], [], [], 0.0
        for b in range(len(PATH_EDGES) - 1):
            m = idx == b
            if m.sum() < 15:
                continue
            k, n = int(y[m].sum()), int(m.sum())
            l, h = wilson(k, n)
            xs.append(q[m].mean()); ys.append(k / n); lo.append(k / n - l); hi.append(h - k / n)
            ece += n * abs(q[m].mean() - k / n)
        ece /= len(y)
        out[name] = dict(ece=ece, mean_pred=float(q.mean()))
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o-", color=c, label=f"{name} (ECE {ece:.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set(xlabel="predicted collision probability of the path", ylabel="observed collision frequency",
           title=f"Path-level reliability, {len(y)} rollouts")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("figures/path_reliability.png", dpi=150)
    summary["path_level"] = out


if __name__ == "__main__":
    synthetic()
    path_level()
    nusc()
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=float)
    for k, v in summary.items():
        print(k)
        print(pd.DataFrame(v).to_string(index=False))
