# TIER: trivial
# Reproduce the reference construction exactly: superpose one blazed grating
# per target spot, each with an independent random piston phase (using the
# instance's own seed), then keep the argument. This is precisely the design
# the grader normalises against, so it scores ~0.1.
import sys, json
import numpy as np


def grating(N, r, c):
    cy = N // 2
    cx = N // 2
    y, x = np.mgrid[0:N, 0:N]
    return 2.0 * np.pi * ((r - cy) * y + (c - cx) * x) / N


inst = json.load(sys.stdin)
N = inst["N"]
spots = [tuple(s) for s in inst["spots"]]
rng = np.random.default_rng(inst["seed"])
field = np.zeros((N, N), dtype=complex)
for (r, c) in spots:
    field += np.exp(1j * (grating(N, r, c) + rng.uniform(0.0, 2.0 * np.pi)))
phase = np.angle(field)
print(json.dumps({"phase": phase.tolist()}))
