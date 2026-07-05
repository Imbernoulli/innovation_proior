# TIER: greedy
# A few Gerchberg-Saxton (iterative Fourier transform) steps from a random
# start: FFT the mask, force the far-field amplitude onto the target spots
# while keeping its phase, IFFT, keep only the aperture phase, repeat. Just a
# handful of iterations already breaks most of the destructive interference
# that plagues the plain superposition reference.
import sys, json
import numpy as np

inst = json.load(sys.stdin)
N = inst["N"]
targets = inst["targets"]
seed = int(inst.get("seed", 0))

T = np.zeros((N, N))
for (ky, kx) in targets:
    T[ky % N, kx % N] = 1.0
T = T / np.sqrt((T ** 2).sum())

rng = np.random.default_rng(seed + 7)
phi = rng.uniform(0.0, 2.0 * np.pi, size=(N, N))
for _ in range(4):
    G = np.fft.fft2(np.exp(1j * phi))
    Gnew = T * np.exp(1j * np.angle(G))
    phi = np.angle(np.fft.ifft2(Gnew))

print(json.dumps({"phase": phi.tolist()}))
