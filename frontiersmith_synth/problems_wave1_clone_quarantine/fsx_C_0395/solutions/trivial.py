# TIER: trivial
# Pass the pre-activation straight through: the IDENTITY (linear) activation. This
# collapses the MLP to a linear classifier -- it reproduces the evaluator's internal
# linear baseline exactly, so every cave maps to ~0.1 and the nonlinear sites under-fit.
import sys, json
import numpy as np

inst = json.load(sys.stdin)
K = int(inst["n_grid"]); lo = float(inst["x_lo"]); hi = float(inst["x_hi"])
grid = np.linspace(lo, hi, K)
vals = grid.copy()
print(json.dumps([float(x) for x in vals]))
