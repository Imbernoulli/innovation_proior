# TIER: greedy
# Plain constant-step gradient descent-ascent (GDA): no momentum, no optimistic term.
# The step is the textbook 1/(2L) with L = largest eigenvalue of the SYMMETRIC part of
# the saddle operator -- i.e. it treats the problem as if it were a plain convex
# quadratic and ignores the rotational (bilinear) coupling A entirely.  Stable and it
# steadily shrinks the residual, but because it neither accelerates nor damps rotation
# it leaves a lot of the T-step budget on the table.
import sys, json
import numpy as np

inst = json.load(sys.stdin)
M = np.array(inst["M"], dtype=float)
T = inst["T"]

S = (M + M.T) / 2.0
L = float(np.linalg.eigvalsh(S).max())
eta = 0.5 / L

print(json.dumps({"a": [eta] * T, "m": [0.0] * T, "o": [0.0] * T}))
