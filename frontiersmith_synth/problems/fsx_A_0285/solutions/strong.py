# TIER: strong
# Shapes the relay density to flatten its autocorrelation, driving the
# congestion constant c(f) well below the uniform value of 2 (typically
# ~1.6-1.75). Deterministic simulated-annealing local search with incremental
# O(n) autocorrelation updates and a few seeded restarts.
import sys
import random
import math


def build_g(f):
    n = len(f)
    g = [0] * (2 * n - 1)
    for a in range(n):
        fa = f[a]
        if fa == 0:
            continue
        for b in range(n):
            fb = f[b]
            if fb:
                g[a + b] += fa * fb
    return g


def optimize(n, seed, mult, scale, cap):
    rng = random.Random(seed)
    f = [scale] * n
    g = build_g(f)
    S = sum(f)

    def cval():
        return 2 * n * max(g) / (S * S) if S > 0 else float("inf")

    best = cval()
    fbest = f[:]
    cur = best
    iters = mult * n
    steps = [-scale // 2, -scale // 4, -16, -4, -1, 1, 4, 16, scale // 4, scale // 2]
    for it in range(iters):
        T = 0.02 * best * (1.0 - it / iters)
        i = rng.randrange(n)
        old = f[i]
        nv = old + rng.choice(steps)
        if nv < 0:
            nv = 0
        elif nv > cap:
            nv = cap
        if nv == old:
            continue
        d = nv - old
        for b in range(n):
            if b == i:
                g[2 * i] += nv * nv - old * old
            else:
                g[i + b] += 2 * d * f[b]
        f[i] = nv
        S += d
        c = cval()
        if c < cur or (T > 0 and rng.random() < math.exp((cur - c) / max(T, 1e-9))):
            cur = c
            if c < best - 1e-15:
                best = c
                fbest = f[:]
        else:
            for b in range(n):
                if b == i:
                    g[2 * i] += old * old - nv * nv
                else:
                    g[i + b] -= 2 * d * f[b]
            f[i] = old
            S -= d
    return fbest, best


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    U = int(toks[1])
    scale = 256
    # keep total work bounded so the largest instance stays well under the limit
    mult = 300 if n <= 96 else 220
    restarts = 3 if n <= 112 else 2

    best_f = None
    best_c = float("inf")
    for s in range(restarts):
        f, c = optimize(n, seed=1000 + s, mult=mult, scale=scale, cap=U)
        if c < best_c:
            best_c = c
            best_f = f

    if best_f is None or sum(best_f) <= 0:
        best_f = [1] * n
    sys.stdout.write(" ".join(map(str, best_f)) + "\n")


if __name__ == "__main__":
    main()
