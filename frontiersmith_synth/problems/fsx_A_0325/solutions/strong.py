# TIER: strong
# Peak-flattening descent on the normalized self-convolution peak.
#
# Uniform intensity is a hard local optimum (c1 = 2); to beat it we minimize a
# soft-max (log-sum-exp) surrogate of the normalized peak with seeded momentum
# projected onto the non-negative orthant, using a few restarts and an
# annealed temperature.  The result is quantized to integers in [0, M] and is
# guarded so it is NEVER worse than the plain uniform show.  Fully deterministic
# (fixed seed, fixed iteration counts -- no wall-clock stopping).
import sys


def c1_int(f, n):
    """Exact-ish float c1 of an integer vector (used only for the final pick)."""
    S = sum(f)
    if S <= 0:
        return float("inf")
    best = 0
    for k in range(2 * n - 1):
        lo = 0 if k < n else k - n + 1
        hi = k if k < n else n - 1
        s = 0
        for i in range(lo, hi + 1):
            s += f[i] * f[k - i]
        if s > best:
            best = s
    return 2.0 * n * best / (S * S)


def quantize(f, M):
    m = max(f)
    if m <= 0:
        return None
    q = [int(round(v / m * M)) for v in f]
    q = [0 if v < 0 else (M if v > M else v) for v in q]
    if max(q) <= 0:
        return None
    return q


def optimize_numpy(n, M):
    import numpy as np
    rng = np.random.default_rng(20240917)
    iters = max(1200, min(9000, 1200000 // n))
    restarts = 3 if n <= 180 else 2
    best_q = None
    best_c = float("inf")
    for r in range(restarts):
        if r == 0:
            f = np.ones(n)
        else:
            f = np.abs(rng.normal(1.0, 0.4, n)) + 0.1
        f = f + 0.05 * rng.standard_normal(n)
        f = np.maximum(f, 1e-3)
        f = f / f.max()
        vel = np.zeros(n)
        for t in range(iters):
            frac = t / iters
            beta = 6.0 + frac * 140.0
            lr = 0.04 * (1.0 - 0.7 * frac)
            S = f.sum()
            g = np.convolve(f, f)
            m = g.max()
            w = np.exp(beta * (g - m))
            w /= w.sum()
            gn = 2.0 * np.correlate(w, f, mode="valid")   # grad of weighted peak
            num = float(np.dot(w, g))
            grad = gn / (S * S) - 2.0 * num / (S ** 3)
            vel = 0.9 * vel - lr * grad
            f = f + vel
            f = np.maximum(f, 1e-4)
            f = f / f.max()
        q = quantize(f.tolist(), M)
        if q is None:
            continue
        c = c1_int(q, n)
        if c < best_c:
            best_c = c
            best_q = q
    return best_q


def main():
    d = sys.stdin.read().split()
    n = int(d[0]); M = int(d[1])

    uniform = [M] * n
    best = uniform
    best_c = c1_int(uniform, n)

    try:
        q = optimize_numpy(n, M)
        if q is not None:
            c = c1_int(q, n)
            if c < best_c:
                best, best_c = q, c
    except Exception:
        # numpy unavailable / any failure -> fall back to the guaranteed
        # uniform schedule (still beats the trivial baseline).
        pass

    sys.stdout.write(" ".join(map(str, best)) + "\n")


main()
