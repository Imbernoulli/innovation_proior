# TIER: strong
# Weighted Gerchberg-Saxton (WGS).  Start from a random-phase superposition, then
# iterate the standard SLM loop: propagate to the far field, read the achieved
# amplitude at each trap, UP-weight the dim traps / DOWN-weight the bright ones,
# reinsert (weight * achieved-phase) at the trap sites, inverse-transform, keep the
# phase.  The feedback drives the traps toward equal depth while concentrating the
# beam's energy into them -- high efficiency AND near-perfect uniformity.  Because
# the illumination is a truncated, rippled Gaussian, a phase-only element still
# cannot reach the ideal Q=1, so the normalized score keeps headroom below 1.0.
import sys, json
import numpy as np

inst = json.load(sys.stdin)
M = inst["M"]
spots = inst["spots"]
seed = int(inst["seed"])
amp = np.array(inst["amp"], dtype=np.float64)

rows = np.array([r for r, c in spots], dtype=int)
cols = np.array([c for r, c in spots], dtype=int)
K = len(spots)

rng = np.random.default_rng(seed)
tgt = np.zeros((M, M), dtype=np.complex128)
tgt[rows, cols] = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=K))
phi = np.angle(np.fft.ifft2(np.fft.ifftshift(tgt)))

w = np.ones(K)
for _ in range(40):
    U = np.fft.fftshift(np.fft.fft2(amp * np.exp(1j * phi)))
    a = np.abs(U[rows, cols])
    mean = a.mean() + 1e-18
    w = w * (mean / (a + 1e-18))
    T = np.zeros((M, M), dtype=np.complex128)
    T[rows, cols] = w * np.exp(1j * np.angle(U[rows, cols]))
    phi = np.angle(np.fft.ifft2(np.fft.ifftshift(T)))

print(json.dumps({"phase": phi.tolist()}))
