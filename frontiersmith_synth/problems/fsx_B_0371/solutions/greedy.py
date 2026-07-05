# TIER: greedy
# Random-phase superposition.  Same superposition-of-gratings construction as the
# baseline, but each target point gets an INDEPENDENT random phase (seeded from the
# public instance seed).  De-correlating the spots kills the coherent interference
# that plagues plain superposition, so efficiency jumps a lot -- but with no
# feedback loop the trap depths are still quite uneven, capping uniformity.
import sys, json
import numpy as np

inst = json.load(sys.stdin)
M = inst["M"]
spots = inst["spots"]
seed = int(inst["seed"])

rng = np.random.default_rng(seed)
tgt = np.zeros((M, M), dtype=np.complex128)
for r, c in spots:
    tgt[r, c] = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi))
phi = np.angle(np.fft.ifft2(np.fft.ifftshift(tgt)))

print(json.dumps({"phase": phi.tolist()}))
