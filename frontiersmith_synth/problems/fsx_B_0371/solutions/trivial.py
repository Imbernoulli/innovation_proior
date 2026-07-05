# TIER: trivial
# Plain superposition of gratings: place a unit, in-phase point at every target
# site in the far-field plane, inverse-transform, and keep only the phase.  This
# reproduces the evaluator's weak analytic baseline, so it scores ~0.1 on every
# instance -- the spots interfere coherently, wrecking both efficiency and
# uniformity.
import sys, json
import numpy as np

inst = json.load(sys.stdin)
M = inst["M"]
spots = inst["spots"]

tgt = np.zeros((M, M), dtype=np.complex128)
for r, c in spots:
    tgt[r, c] = 1.0
phi = np.angle(np.fft.ifft2(np.fft.ifftshift(tgt)))

print(json.dumps({"phase": phi.tolist()}))
