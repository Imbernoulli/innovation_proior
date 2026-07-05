# TIER: greedy
# "Just tune the learning rate."  Keep the relay gains CONSTANT across all legs but pick the
# single best constant step by simulating the whole T-leg run over a geometric grid (using the
# optimistic template a=b=eta, which is stable for these monotone operators).  This beats the
# untuned conservative reference, but a fixed step is bottlenecked by the operator's worst
# eigen-direction, so it stays well short of a time-varying schedule.
import sys, json
import numpy as np

inst = json.load(sys.stdin)
M = np.array(inst["M"], dtype=float)
q = np.array(inst["q"], dtype=float)
z0 = np.array(inst["z0"], dtype=float)
T = inst["T"]


def final_norm(eta):
    z = z0.copy()
    g = M @ z + q
    gprev = g.copy()
    for _ in range(T):
        z = z - eta * g - eta * (g - gprev)
        gprev = g
        g = M @ z + q
        if not np.all(np.isfinite(g)):
            return float("inf")
    return float(np.linalg.norm(g))


best_eta, best_val = 0.0, float("inf")
for eta in np.geomspace(1e-4, 2.0, 80):
    v = final_norm(float(eta))
    if np.isfinite(v) and v < best_val:
        best_val, best_eta = v, float(eta)

print(json.dumps({"a": [best_eta] * T, "b": [best_eta] * T}))
