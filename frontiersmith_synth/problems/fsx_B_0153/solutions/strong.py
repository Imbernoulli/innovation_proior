# TIER: strong
# Weighted Gerchberg-Saxton (the "adaptive-additive" / GAA scheme used for real SLM
# spot-array holograms). Start from the analog superposition phase, then alternate:
# propagate to the far field, keep the far-field phase on the spots but overwrite their
# amplitudes with adaptive per-spot targets, zero everything else, propagate back, and
# keep only the aperture-plane phase (phase-only SLM constraint). Between iterations the
# per-spot target weights are re-weighted to pull power OUT of the spots that came out
# too bright and INTO the dim ones -- directly attacking the harsh min/max uniformity
# term while GS holds efficiency high. Many iterations give a far better
# efficiency*uniformity composite than the binary baseline or the one-shot superposition.
import sys, json
import numpy as np

inst = json.load(sys.stdin)
N = inst["N"]
targets = inst["targets"]

y = np.arange(N).reshape(N, 1)
x = np.arange(N).reshape(1, N)
acc = np.zeros((N, N), dtype=complex)
for (r, c) in targets:
    acc += np.exp(2j * np.pi * ((r - N // 2) * y + (c - N // 2) * x) / N)
phase = np.angle(acc)

rows = np.array([t[0] for t in targets])
cols = np.array([t[1] for t in targets])
M = len(targets)
w = np.ones(M)                       # adaptive per-spot target weights

for it in range(50):
    G = np.fft.fftshift(np.fft.fft2(np.exp(1j * phase)))
    amp = np.abs(G[rows, cols])
    ph = np.angle(G[rows, cols])
    if it > 0 and np.all(amp > 0):   # adaptive-additive: boost dim spots, damp bright
        w = w * (np.mean(amp) / amp) ** 0.7
        w = w / np.mean(w)
    Gp = np.zeros((N, N), dtype=complex)
    Gp[rows, cols] = w * np.exp(1j * ph)
    phase = np.angle(np.fft.ifft2(np.fft.ifftshift(Gp)))

print(json.dumps({"phase": phase.tolist()}))
