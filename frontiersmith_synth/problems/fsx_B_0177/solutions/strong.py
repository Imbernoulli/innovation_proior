# TIER: strong
# Weighted / adaptive-additive Gerchberg-Saxton (Di Leonardo-style feedback). Same
# spot-array GS backbone as greedy (far-field target = amplitude on patches, zero
# elsewhere), but each iteration reweights the clamped patch amplitudes by the measured
# deviation from the prescribed profile: patches that run bright get pulled down, dim
# ones pushed up, so the achieved intensity converges to I_i ~ w_i while efficiency
# stays high. High efficiency AND high profile fidelity.
import sys, json
import numpy as np

inst = json.load(sys.stdin)
N = inst["N"]
targets = inst["targets"]
weights = inst["weights"]

rows = np.array([t[0] for t in targets], dtype=int)
cols = np.array([t[1] for t in targets], dtype=int)
g = np.sqrt(np.array(weights, dtype=float))     # desired relative amplitudes

# start from the grating superposition
y = np.arange(N).reshape(N, 1)
x = np.arange(N).reshape(1, N)
acc = np.zeros((N, N), dtype=complex)
for (r, c), w in zip(targets, weights):
    acc += np.sqrt(float(w)) * np.exp(2j * np.pi * ((r - N // 2) * y + (c - N // 2) * x) / N)
phase = np.angle(acc)

a = g.copy()                                    # working target amplitudes
for _ in range(60):
    E = np.exp(1j * phase)
    Gs = np.fft.fftshift(np.fft.fft2(E))
    amp = np.abs(Gs[rows, cols])
    ph = np.angle(Gs[rows, cols])
    # feedback: m_i = amp_i / g_i should be equal across patches when on-profile.
    m = amp / np.maximum(g, 1e-12)
    mbar = float(np.mean(m))
    a = a * (mbar / np.maximum(m, 1e-12))       # pull each patch toward the mean ratio
    a = np.maximum(a, 1e-9)
    T = np.zeros((N, N), dtype=float)
    T[rows, cols] = a
    B = T * np.exp(1j * np.angle(Gs))
    E2 = np.fft.ifft2(np.fft.ifftshift(B))
    phase = np.angle(E2)

print(json.dumps({"phase": phase.tolist()}))
