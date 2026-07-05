# TIER: strong
# Multi-restart weighted Gerchberg-Saxton. Run the iterative Fourier-transform
# algorithm from several deterministic random starts; within each run use an
# adaptive-additive weight per spot that boosts whichever ablation spots are
# currently dimmest, driving the array toward equal brightness AND high
# efficiency. Score every candidate mask with the exact grader metric and keep
# the best one found. Beats the reference DOE substantially, with headroom left.
import sys, json
import numpy as np

inst = json.load(sys.stdin)
N = inst["N"]
targets = inst["targets"]
seed = int(inst.get("seed", 0))
idx = [(ky % N, kx % N) for (ky, kx) in targets]


def metric(phi):
    G = np.fft.fft2(np.exp(1j * phi))
    I = np.abs(G) ** 2
    P = I.sum()
    spot = np.array([I[a, b] for (a, b) in idx])
    eta = spot.sum() / P
    mx, mn = spot.max(), spot.min()
    U = 1.0 - (mx - mn) / (mx + mn + 1e-30)
    return eta * U


def run(start_seed, iters, adaptive):
    T0 = np.zeros((N, N))
    for (a, b) in idx:
        T0[a, b] = 1.0
    T0 = T0 / np.sqrt((T0 ** 2).sum())
    w = np.ones(len(idx))
    rng = np.random.default_rng(start_seed)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=(N, N))
    for it in range(iters):
        G = np.fft.fft2(np.exp(1j * phi))
        amp = np.abs(G)
        if adaptive and it > 0:
            spot = np.array([amp[a, b] for (a, b) in idx])
            avg = spot.mean() + 1e-30
            # push dim spots up: increase target weight where current amp < mean
            w = w * (avg / (spot + 1e-30)) ** 0.5
            w = np.clip(w, 0.2, 5.0)
        Tw = np.zeros((N, N))
        for (a, b), wi in zip(idx, w):
            Tw[a, b] = wi
        s = np.sqrt((Tw ** 2).sum())
        Tw = Tw / (s if s > 0 else 1.0)
        Gnew = Tw * np.exp(1j * np.angle(G))
        phi = np.angle(np.fft.ifft2(Gnew))
    return phi


best = None
best_m = -1.0
for r in range(6):
    for adap in (False, True):
        phi = run(1000 + 17 * r + (1 if adap else 0), 60, adap)
        m = metric(phi)
        if m > best_m:
            best_m = m
            best = phi

print(json.dumps({"phase": best.tolist()}))
