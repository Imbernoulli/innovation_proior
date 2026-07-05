# TIER: greedy
import sys, json
import numpy as np
inst = json.load(sys.stdin)
N = inst["N"]; targets = inst["targets"]
idx = np.array(targets)
yy, xx = np.mgrid[0:N, 0:N]
f = np.zeros((N, N), dtype=complex)
for (r, c) in targets:
    ky = r - N // 2; kx = c - N // 2
    f += np.exp(1j * (2 * np.pi * (ky * yy + kx * xx) / N))
phase = np.angle(f)
w = np.ones(len(targets))
for _ in range(5):                       # a few weighted Gerchberg-Saxton passes
    U = np.exp(1j * phase)
    G = np.fft.fftshift(np.fft.fft2(U))
    amp = np.abs(G[idx[:, 0], idx[:, 1]])
    w = w * (amp.mean() / np.maximum(amp, 1e-9)); w = w / w.mean()
    ph = np.angle(G[idx[:, 0], idx[:, 1]])
    Gn = np.zeros((N, N), dtype=complex)
    Gn[idx[:, 0], idx[:, 1]] = w * np.exp(1j * ph)
    phase = np.angle(np.fft.ifft2(np.fft.ifftshift(Gn)))
print(json.dumps({"phase": phase.tolist()}))
