# TIER: strong
# Weighted Gerchberg-Saxton / iterative Fourier-transform algorithm (IFTA).
# Start from a random-phase superposition, then alternate between pupil and
# far-field planes: in the far field keep the CURRENT phase but reset the
# amplitude at the target spots to an ADAPTIVELY-REWEIGHTED value (spots that
# came out dim are boosted next round), and null everything else; back-transform
# and re-impose the phase-only pupil constraint. This drives uniformity toward
# ~1 and packs efficiency near the phase-only limit -- far above the baseline,
# yet still short of the (unreachable) M==1 ceiling.
import sys, json
import numpy as np

inst = json.load(sys.stdin)
N = inst["N"]
targets = inst["targets"]
gamma = inst["gamma"]
K = len(targets)
idx = np.asarray(targets, dtype=int)
ty, tx = idx[:, 0], idx[:, 1]
y = np.arange(N).reshape(-1, 1)
x = np.arange(N).reshape(1, -1)


def obj(phi):
    P = np.abs(np.fft.fft2(np.exp(1j * phi))) ** 2
    I = P[ty, tx]
    eta = I.sum() / P.sum()
    u = 1.0 - (I.max() - I.min()) / (I.max() + I.min() + 1e-30)
    return eta * (u ** gamma)


def run(seed, iters):
    rng = np.random.default_rng(seed)
    C = np.zeros((N, N), dtype=complex)
    ph = rng.uniform(0.0, 2 * np.pi, K)
    for a, (fy, fx) in zip(ph, targets):
        C += np.exp(1j * a) * np.exp(2j * np.pi * (fy * y + fx * x) / N)
    phi = np.angle(C)
    w = np.ones(K)
    best = phi
    best_v = obj(phi)
    for _ in range(iters):
        F = np.fft.fft2(np.exp(1j * phi))
        P = np.abs(F) ** 2
        I = P[ty, tx]
        Im = I.mean() + 1e-30
        w = w * (Im / (I + 1e-30)) ** 0.5     # boost the dim spots
        Ft = np.zeros((N, N), dtype=complex)
        Ft[ty, tx] = w * np.exp(1j * np.angle(F[ty, tx]))
        phi = np.angle(np.fft.ifft2(Ft))
        v = obj(phi)
        if v > best_v:
            best_v = v
            best = phi
    return best, best_v


best_phi = None
best_v = -1.0
for s in range(3):
    phi, v = run(90000 + 17 * s, 45)
    if v > best_v:
        best_v = v
        best_phi = phi

print(json.dumps({"phase": best_phi.tolist()}))
