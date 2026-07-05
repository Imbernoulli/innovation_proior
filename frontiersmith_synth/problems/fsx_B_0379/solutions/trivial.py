# TIER: trivial
# Superposition-of-gratings DOE: sum one blazed grating per target spot and
# take the phase. This is exactly the grader's reference construction, so it
# scores ~0.1. It splits the beam but the gratings interfere, so some ablation
# spots come out much dimmer than others (poor uniformity).
import sys, json
import numpy as np

inst = json.load(sys.stdin)
N = inst["N"]
targets = inst["targets"]
y, x = np.mgrid[0:N, 0:N]
acc = np.zeros((N, N), dtype=complex)
for (ky, kx) in targets:
    acc += np.exp(1j * 2.0 * np.pi * (ky * y + kx * x) / N)
phi = np.angle(acc)
print(json.dumps({"phase": phi.tolist()}))
