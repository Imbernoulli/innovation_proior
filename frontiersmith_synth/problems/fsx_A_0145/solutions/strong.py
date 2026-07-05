# TIER: strong
"""Randomised coordinate-descent search (fixed seed -> deterministic output)
that shapes the emission profile to flatten the self-convolution, pushing c1
well below the constant (2.0) and triangle (>2.3) baselines toward ~1.58.

The self-convolution is maintained INCREMENTALLY: changing one coordinate
touches only O(n) convolution entries, so each move is O(n) instead of O(n^2).
"""
import sys
import random


def full_conv(f):
    n = len(f)
    conv = [0.0] * (2 * n - 1)
    for a in range(n):
        fa = f[a]
        if fa == 0.0:
            continue
        for b in range(n):
            conv[a + b] += fa * f[b]
    return conv


def optimise(n, V):
    rng = random.Random(777)
    iters = 4000 + 200 * n
    best = None
    bestv = 1e18

    starts = [[1.0] * n]
    for a in (0.8, 1.0, 2.0):
        s = []
        for i in range(n):
            x = 2.0 * i / (n - 1) - 1.0 if n > 1 else 0.0
            s.append(1.0 + a * x * x)
        starts.append(s)

    for s0 in starts:
        f = [max(v, 1e-3) for v in s0]
        conv = full_conv(f)
        S = sum(f)
        curv = 2 * n * max(conv) / (S * S)
        loc = f[:]
        locv = curv
        for it in range(iters):
            i = rng.randrange(n)
            scale = 0.3 * (1.0 - it / iters) + 0.01
            nv = f[i] + rng.gauss(0.0, scale)
            if nv < 0.0:
                nv = 0.0
            old = f[i]
            delta = nv - old
            if delta == 0.0:
                continue
            # incremental conv update
            twod = 2.0 * delta
            for j in range(n):
                if j != i:
                    conv[i + j] += twod * f[j]
            conv[2 * i] += nv * nv - old * old
            f[i] = nv
            S += delta
            v = 2 * n * max(conv) / (S * S)
            if v < curv or rng.random() < 0.02 * (1.0 - it / iters):
                curv = v
                if v < locv:
                    loc = f[:]
                    locv = v
            else:
                # revert
                for j in range(n):
                    if j != i:
                        conv[i + j] -= twod * f[j]
                conv[2 * i] -= nv * nv - old * old
                f[i] = old
                S -= delta
        if locv < bestv:
            bestv = locv
            best = loc

    mx = max(best)
    target = min(1000, V)
    ints = [max(0, int(round(v / mx * target))) for v in best]
    if sum(ints) <= 0:
        ints[0] = 1
    return ints


def main():
    tok = sys.stdin.read().split()
    n = int(tok[0]); V = int(tok[1])
    f = optimise(n, V)
    sys.stdout.write(" ".join(map(str, f)) + "\n")


if __name__ == "__main__":
    main()
