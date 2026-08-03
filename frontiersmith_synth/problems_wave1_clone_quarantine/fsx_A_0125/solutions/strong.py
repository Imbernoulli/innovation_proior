# TIER: strong
# Shape the gain profile to flatten the leakage spectrum and cut its total L2
# energy below the uniform value. Projected Adam gradient descent on
#     E(f) = n * sum_k c_k^2 / (sum f)^4 ,  c = f * f (autoconvolution),
# with several deterministic restarts, projected to non-negative gains.
# Gradient (exact): dE/df = n * (dP*S - 4*P) / S^5 with dP = 4 * (c corr f),
# S = sum f, P = sum_k c_k^2.
import sys

try:
    import numpy as np
    HAVE_NP = True
except Exception:
    HAVE_NP = False


def energy(f, n):
    S = f.sum()
    c = np.convolve(f, f)
    return n * (c * c).sum() / (S ** 4)


def optimize(n):
    best_val = 1e18
    best_f = np.ones(n)
    for seed in range(5):
        rng = np.random.default_rng(seed)
        f = rng.random(n) + 0.3
        m = np.zeros(n)
        v = np.zeros(n)
        b1, b2, lr = 0.9, 0.999, 0.02
        for t in range(1500):
            S = f.sum()
            c = np.convolve(f, f)          # length 2n-1
            P = float((c * c).sum())
            g = np.correlate(c, f, "valid")  # length n: g_j = sum_k c_k f_{k-j}
            dP = 4.0 * g
            grad = n * (dP * S - 4.0 * P) / (S ** 5)
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
        # numpy-free fallback: symmetric tapered "smile" profile (denser at the
        # ends) still spreads leakage better than uniform.
        out = [1.0 + 1.2 * abs(2.0 * i / (n - 1) - 1.0) for i in range(n)]
    print(" ".join("%.6f" % x for x in out))


if __name__ == "__main__":
    main()
