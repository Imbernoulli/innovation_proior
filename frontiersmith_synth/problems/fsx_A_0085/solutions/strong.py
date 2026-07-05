# TIER: strong
# Shape the habitat profile to flatten the autoconvolution peak. A smooth
# min-max descent: soft-max weighting of the encounter spectrum + Adam steps on
# the density profile, projected to be non-negative, with several deterministic
# restarts. Drives the index below the uniform value (~2.0) toward ~1.55.
import sys

try:
    import numpy as np
    HAVE_NP = True
except Exception:
    HAVE_NP = False


def objective(f, n):
    s = f.sum()
    c = np.convolve(f, f)
    return 2.0 * n * c.max() / (s * s)


def optimize(n):
    best_val = 1e18
    best_f = np.ones(n)
    for seed in range(6):
        rng = np.random.default_rng(seed)
        f = rng.random(n) + 0.3
        m = np.zeros(n)
        v = np.zeros(n)
        b1, b2, lr = 0.9, 0.999, 0.02
        iters = 1200
        for t in range(iters):
            beta = 20.0 + t * 0.06
            s = f.sum()
            c = np.convolve(f, f)
            cm = c.max()
            w = np.exp(beta * (c - cm))
            w /= w.sum()
            # dM/df_j = 2 * sum_k w_k f_{k-j} = 2 * dot(w[j:j+n], f)
            g = np.empty(n)
            for j in range(n):
                g[j] = 2.0 * np.dot(w[j:j + n], f)
            M = float((w * c).sum())
            grad = 2.0 * n * (g * s * s - M * 2.0 * s) / (s ** 4)
            m = b1 * m + (1 - b1) * grad
            v = b2 * v + (1 - b2) * grad * grad
            mh = m / (1 - b1 ** (t + 1))
            vh = v / (1 - b2 ** (t + 1))
            f = f - lr * mh / (np.sqrt(vh) + 1e-9)
            f = np.maximum(f, 1e-6)
        val = objective(f, n)
        if val < best_val:
            best_val = val
            best_f = f.copy()
    return best_f


def main():
    n = int(sys.stdin.read().split()[0])
    if HAVE_NP:
        f = optimize(n)
        out = [float(x) for x in f]
    else:
        # fallback: symmetric "smile" profile (higher at the ends) still beats uniform
        out = [1.0 + 1.5 * abs(2.0 * i / (n - 1) - 1.0) for i in range(n)]
    print(" ".join("%.6f" % x for x in out))


if __name__ == "__main__":
    main()
