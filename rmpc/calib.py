"""Calibration metrics and temperature scaling for occupancy probabilities."""
import numpy as np
from scipy.optimize import minimize_scalar


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def apply_temperature(p, T):
    return sigmoid(logit(p) / T)


def fit_temperature(p, y, bounds=(0.05, 20.0)):
    """Temperature T minimizing the binary NLL of sigmoid(logit(p) / T) against labels y."""
    z, y = logit(np.asarray(p, float)), np.asarray(y, float)

    def nll(logT):
        q = np.clip(sigmoid(z / np.exp(logT)), 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(q) + (1 - y) * np.log1p(-q))

    r = minimize_scalar(nll, bounds=np.log(bounds), method="bounded")
    return float(np.exp(r.x))


def reliability(p, y, n_bins=10, edges=None):
    """Per-bin mean prediction, observed frequency and count. Bins with no samples are NaN."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, n_bins + 1) if edges is None else np.asarray(edges)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    conf = np.full(len(edges) - 1, np.nan)
    freq = np.full(len(edges) - 1, np.nan)
    count = np.bincount(idx, minlength=len(edges) - 1)
    for b in range(len(edges) - 1):
        if count[b]:
            conf[b], freq[b] = p[idx == b].mean(), y[idx == b].mean()
    return conf, freq, count


def ece(p, y, n_bins=10, edges=None):
    conf, freq, count = reliability(p, y, n_bins, edges)
    ok = count > 0
    return float(np.sum(count[ok] * np.abs(conf[ok] - freq[ok])) / count.sum())


def wilson(k, n, z=1.96):
    """95% Wilson interval for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)
