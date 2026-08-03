# TIER: strong
# Minimize the L2 spectral energy directly. Deterministic Adam descent on the
# capacity profile with the analytic gradient of
#     L(f) = n * sum_k c_k^2 / (sum f)^4 ,
# projected to be non-negative, with several seeded restarts. This shapes a
# tapered profile that drives the energy below the uniform value toward ~0.58.
import sys

try:
    import numpy as np
    HAVE_NP = True
except Exception:
    HAVE_NP = False


def energy(f, n):
    s = f.sum()
    c = np.convolve(f, f)
    return n * np.sum(c * c) / s ** 4


def optimize(n):
    best_val = 1e18
    best_f = np.ones(n)
    iters = 1200 if n <= 160 else 700
    for seed in range(4):
        rng = np.random.default_rng(seed)
        f = rng.random(n) + 0.3
        m = np.zeros(n)
        v = np.zeros(n)
        b1, b2, lr = 0.9, 0.999, 0.03
        for t in range(iters):
            s = f.sum()
            c = np.convolve(f, f)
            # d(sum c^2)/df_j = 4 * sum_k c_k f_{k-j} = 4 * correlate(c, f)[j]
            g = 4.0 * np.correlate(c, f, "valid")
            S2 = np.sum(c * c)
            grad = n * (g * s ** 4 - S2 * 4.0 * s ** 3) / s ** 8
            m = b1 * m + (1 - b1) * grad
            v = b2 * v + (1 - b2) * grad * grad
            mh = m / (1 - b1 ** (t + 1))
            vh = v / (1 - b2 ** (t + 1))
            f = f - lr * mh / (np.sqrt(vh) + 1e-9)
            f = np.maximum(f, 1e-6)
        val = energy(f, n)
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
        # fallback: analytic tapered 'tent' profile (denser at the ends) still
        # beats uniform on the L2 energy.
        out = [1.0 + 1.5 * abs(2.0 * i / (n - 1) - 1.0) for i in range(n)]
    print(" ".join("%.6f" % x for x in out))


if __name__ == "__main__":
    main()
