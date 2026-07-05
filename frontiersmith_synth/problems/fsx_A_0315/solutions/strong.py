# TIER: strong
# Minimize the peak self-convolution c1 within the ceiling box by projected
# subgradient descent. Starts from the full-ceiling profile and repeatedly nudges
# mass away from the current glare-peak offset, staying inside [0, u_i]. This
# flattens the autocorrelation below the uniform value (c1 ~ 1.6, vs 2.0 uniform),
# with the exact optimum left open. Deterministic (seeded, fixed schedule).
import sys
import numpy as np


def c1(f, N):
    g = np.convolve(f, f)
    s = f.sum()
    return 2.0 * N * g.max() / (s * s) if s > 0 else 9e18


def main():
    toks = sys.stdin.read().split()
    N = int(toks[0])
    caps = np.array([float(t) for t in toks[1:1 + N]], dtype=float)

    f = caps.copy()
    best = f.copy()
    bestv = c1(f, N)

    iters = 3000
    step = 0.08
    ar = np.arange(N)
    for _ in range(iters):
        g = np.convolve(f, f)
        k = int(g.argmax())
        s = f.sum()
        gk = g[k]
        idx = k - ar
        m = (idx >= 0) & (idx < N)
        grad = np.zeros(N)
        grad[m] = 2.0 * f[idx[m]]
        # gradient of c1 = 2N * g_k / s^2
        og = 2.0 * N * (grad * s * s - gk * 2.0 * s) / (s ** 4)
        nrm = np.abs(og).max() + 1e-12
        f = f - step * og / nrm
        f = np.clip(f, 0.0, caps)
        v = c1(f, N)
        if v < bestv:
            bestv = v
            best = f.copy()

    # guard against a degenerate all-zero drift
    if best.sum() <= 1e-9:
        best = caps.copy()

    sys.stdout.write(" ".join("%.6f" % x for x in best) + "\n")


if __name__ == "__main__":
    main()
