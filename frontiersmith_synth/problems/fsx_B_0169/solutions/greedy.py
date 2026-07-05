# TIER: greedy
# Plain (unweighted) Gerchberg-Saxton / IFTA: start from the random-grating
# superposition, then iterate FFT -> replace target amplitudes with a uniform
# value (keeping the phase) -> IFFT -> keep the phase, snapping to the device
# levels each round. This tightens the pattern somewhat but, lacking adaptive
# spot weighting, it does not fully equalise brightness on the harder boards.
import sys, json
import numpy as np


def grating(N, r, c):
    cy = N // 2
    cx = N // 2
    y, x = np.mgrid[0:N, 0:N]
    return 2.0 * np.pi * ((r - cy) * y + (c - cx) * x) / N


def quantize(phase, L):
    step = 2.0 * np.pi / L
    return (np.round(phase / step) * step) % (2.0 * np.pi)


inst = json.load(sys.stdin)
N = inst["N"]
L = inst["L"]
spots = [tuple(s) for s in inst["spots"]]

rng = np.random.default_rng(inst["seed"])
field = np.zeros((N, N), dtype=complex)
for (r, c) in spots:
    field += np.exp(1j * (grating(N, r, c) + rng.uniform(0.0, 2.0 * np.pi)))
phase = np.angle(field)

for _ in range(30):
    q = quantize(phase, L)
    f = np.fft.fftshift(np.fft.fft2(np.exp(1j * q)))
    ang = np.angle(f)
    newf = np.zeros((N, N), dtype=complex)
    for (r, c) in spots:
        newf[r, c] = np.exp(1j * ang[r, c])
    g = np.fft.ifft2(np.fft.ifftshift(newf))
    phase = np.angle(g)

print(json.dumps({"phase": phase.tolist()}))
