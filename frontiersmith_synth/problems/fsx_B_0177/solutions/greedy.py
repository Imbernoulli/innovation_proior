# TIER: greedy
# Plain Gerchberg-Saxton toward the weighted profile: start from the grating
# superposition, then iterate the phase-only <-> far-field constraint. The far-field
# target amplitude is sqrt(w_i) on each patch and ZERO everywhere else (standard
# spot-array GS), so energy is concentrated onto the corridor. WITHOUT feedback
# weighting the achieved profile drifts (bright patches stay ahead), so fidelity is
# only moderate -- it beats the naive baseline but leaves headroom.
import sys, json
import numpy as np

inst = json.load(sys.stdin)
N = inst["N"]
targets = inst["targets"]
weights = inst["weights"]

rows = np.array([t[0] for t in targets], dtype=int)
cols = np.array([t[1] for t in targets], dtype=int)
g = np.sqrt(np.array(weights, dtype=float))     # desired relative amplitudes

# fixed far-field target amplitude: sqrt(w_i) on patches, 0 elsewhere
T = np.zeros((N, N), dtype=float)
T[rows, cols] = g

# start from the grating superposition
y = np.arange(N).reshape(N, 1)
x = np.arange(N).reshape(1, N)
acc = np.zeros((N, N), dtype=complex)
for (r, c), w in zip(targets, weights):
    acc += np.sqrt(float(w)) * np.exp(2j * np.pi * ((r - N // 2) * y + (c - N // 2) * x) / N)
phase = np.angle(acc)

for _ in range(25):
    E = np.exp(1j * phase)
    Gs = np.fft.fftshift(np.fft.fft2(E))
    B = T * np.exp(1j * np.angle(Gs))           # clamp amplitudes to target, keep phase
    E2 = np.fft.ifft2(np.fft.ifftshift(B))
    phase = np.angle(E2)

print(json.dumps({"phase": phase.tolist()}))
