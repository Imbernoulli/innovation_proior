# TIER: greedy
# Random-phase superposition: give each single-spot grating an independent
# random phase offset (the classic trick that de-correlates interference and
# flattens the fan-out), and keep the best of several random draws. Beats the
# in-phase baseline on both efficiency and uniformity, but stagnates far below
# a true iterative Fourier-transform design.
import sys, json
import numpy as np

inst = json.load(sys.stdin)
N = inst["N"]
targets = inst["targets"]
gamma = inst["gamma"]
K = len(targets)
y = np.arange(N).reshape(-1, 1)
x = np.arange(N).reshape(1, -1)
gratings = [np.exp(2j * np.pi * (fy * y + fx * x) / N) for fy, fx in targets]
idx = np.asarray(targets, dtype=int)


def obj(phi):
    P = np.abs(np.fft.fft2(np.exp(1j * phi))) ** 2
    I = P[idx[:, 0], idx[:, 1]]
    eta = I.sum() / P.sum()
    u = 1.0 - (I.max() - I.min()) / (I.max() + I.min() + 1e-30)
    return eta * (u ** gamma)


best_phi = None
best_v = -1.0
for t in range(8):
    rng = np.random.default_rng(20260701 + 13 * t)
    ph = rng.uniform(0.0, 2 * np.pi, K)
    C = np.zeros((N, N), dtype=complex)
    for g, a in zip(gratings, ph):
        C += np.exp(1j * a) * g
    phi = np.angle(C)
    v = obj(phi)
    if v > best_v:
        best_v = v
        best_phi = phi

print(json.dumps({"phase": best_phi.tolist()}))
