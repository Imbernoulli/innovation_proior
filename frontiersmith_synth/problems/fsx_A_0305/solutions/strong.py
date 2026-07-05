# TIER: strong
# Projected-gradient descent that directly minimises the self-convolution peak
# c1(f) = 2n * max_k conv(f,f)_k / (sum f)^2. Deterministic: the random restarts
# are seeded only from n, so the profile (and hence the per-test score) depends on
# the instance. Reaches c1 ~= 1.55-1.65, well below both baseline and the flat
# profile. The true optimum is unknown, so this leaves headroom.
import sys
import random


def conv(f):
    n = len(f)
    out = [0.0] * (2 * n - 1)
    for i in range(n):
        fi = f[i]
        for j in range(n):
            out[i + j] += fi * f[j]
    return out


def c1(f):
    n = len(f)
    s = sum(f)
    return 2.0 * n * max(conv(f)) / (s * s)


def optimise(n, seed, restarts=15, iters=900, lr=0.02):
    rng = random.Random(seed)
    best_val = None
    best_f = None
    for _ in range(restarts):
        f = [rng.random() + 0.1 for _ in range(n)]
        for _ in range(iters):
            c = conv(f)
            s = sum(f)
            # argmax of the self-convolution
            k = 0
            cm = c[0]
            for idx in range(1, len(c)):
                if c[idx] > cm:
                    cm = c[idx]
                    k = idx
            ck = c[k]
            g = [0.0] * n
            for i in range(n):
                j = k - i
                if 0 <= j < n:
                    g[i] += 2.0 * f[j]
            s2 = s * s
            s4 = s2 * s2
            grad = [2.0 * n * (g[i] * s2 - ck * 2.0 * s) / s4 for i in range(n)]
            f = [max(1e-9, f[i] - lr * grad[i]) for i in range(n)]
        v = c1(f)
        if best_val is None or v < best_val:
            best_val = v
            best_f = f
    return best_f


def main():
    n = int(sys.stdin.read().split()[0])
    f = optimise(n, 7919 + n)
    print(" ".join("%.10f" % x for x in f))


if __name__ == "__main__":
    main()
