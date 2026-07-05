# TIER: trivial
import sys, json
import numpy as np
inst = json.load(sys.stdin)
N = inst["N"]; targets = inst["targets"]
yy, xx = np.mgrid[0:N, 0:N]
f = np.zeros((N, N), dtype=complex)
for (r, c) in targets:
    ky = r - N // 2; kx = c - N // 2
    f += np.exp(1j * (2 * np.pi * (ky * yy + kx * xx) / N))
phase = np.angle(f)
print(json.dumps({"phase": phase.tolist()}))
