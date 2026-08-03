# TIER: strong
# A smooth, self-gated GELU-style activation: a(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x +
# 0.044715*x^3))). It is nonlinear (unlocks the hidden layer like ReLU), but its smooth,
# soft-gated shape passes a small signal/gradient for slightly-negative pre-activations
# and curves gently near the origin -- which under the deliberately hot base LR keeps
# training stable and fits the noisier caves better than a hard rectifier. It grows
# ~linearly for large positive inputs (no runaway slope), so it generalizes across every
# survey site -> high geometric mean.
import sys, json
import numpy as np

inst = json.load(sys.stdin)
K = int(inst["n_grid"]); lo = float(inst["x_lo"]); hi = float(inst["x_hi"])
grid = np.linspace(lo, hi, K)
c = (2.0 / np.pi) ** 0.5
vals = 0.5 * grid * (1.0 + np.tanh(c * (grid + 0.044715 * grid ** 3)))
print(json.dumps([float(x) for x in vals]))
