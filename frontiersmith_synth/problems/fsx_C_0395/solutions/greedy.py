# TIER: greedy
# The obvious fix: add a nonlinearity -- a plain ReLU. This unlocks the hidden layer so
# the network can carve the nonlinear caves, a big jump over the linear baseline. But the
# hard rectifier (zero response / zero gradient for negative pre-activations) leaves per-
# site headroom on the noisier caves, so a smoother activation can still do better.
import sys, json
import numpy as np

inst = json.load(sys.stdin)
K = int(inst["n_grid"]); lo = float(inst["x_lo"]); hi = float(inst["x_hi"])
grid = np.linspace(lo, hi, K)
vals = np.maximum(0.0, grid)
print(json.dumps([float(x) for x in vals]))
