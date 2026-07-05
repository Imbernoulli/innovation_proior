# TIER: strong
# Weighted Gerchberg-Saxton (weighted IFTA), the standard high-performance
# spot-array generator. Same loop as greedy, but each target spot carries an
# adaptive weight updated as w *= sqrt(mean_intensity / spot_intensity) so
# persistently dim spots are pushed harder. Snapping to the device levels
# every iteration keeps the design robust to the binary quantisation. This
# equalises the constellation (uniformity -> ~1) while holding efficiency,
# substantially beating the reference superposition.
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
M = len(spots)

rng = np.random.default_rng(inst["seed"])
field = np.zeros((N, N), dtype=complex)
for (r, c) in spots:
    field += np.exp(1j * (grating(N, r, c) + rng.uniform(0.0, 2.0 * np.pi)))
phase = np.angle(field)

w = np.ones(M)
for _ in range(80):
    q = quantize(phase, L)
    f = np.fft.fftshift(np.fft.fft2(np.exp(1j * q)))
    ang = np.angle(f)
    amp = np.abs(f)
    Ik = np.array([amp[r, c] ** 2 for (r, c) in spots], dtype=float)
    mean = Ik.mean() + 1e-12
    w = w * np.sqrt(mean / (Ik + 1e-12))
    w = w / w.mean()
    newf = np.zeros((N, N), dtype=complex)
    for i, (r, c) in enumerate(spots):
        newf[r, c] = w[i] * np.exp(1j * ang[r, c])
    g = np.fft.ifft2(np.fft.ifftshift(newf))
    phase = np.angle(g)

print(json.dumps({"phase": phase.tolist()}))
