# TIER: strong
"""Optimize the density profile to minimize the peak self-overlap.

Start from a flat water-filled profile, then run seeded coordinate-descent:
repeatedly move a little mass from one pool to another (keeping the total and
the capacities satisfied) whenever it lowers c1 = 2n*max(conv(f,f))/(sum f)^2.
The optimum is edge-heavy (spread away from the centre), reaching c1 well
below the flat value of 2.  Deterministic (fixed RNG seed)."""
import sys, random

try:
    import numpy as np
    def peak(f):
        g = np.convolve(f, f)
        return float(g.max())
    HAVE_NP = True
except Exception:  # pragma: no cover
    HAVE_NP = False
    def peak(f):
        n = len(f); best = 0.0
        for k in range(2 * n - 1):
            lo = max(0, k - (n - 1)); hi = min(k, n - 1)
            s = 0.0
            for i in range(lo, hi + 1):
                s += f[i] * f[k - i]
            if s > best:
                best = s
        return best


def waterfill(cap, S):
    n = len(cap); f = [0.0] * n; active = [True] * n; r = float(S)
    for _ in range(n + 5):
        na = sum(1 for a in active if a)
        if r <= 1e-12 or na == 0:
            break
        lvl = r / na
        hit = [i for i in range(n) if active[i] and (cap[i] - f[i]) < lvl]
        if hit:
            for i in hit:
                r -= (cap[i] - f[i]); f[i] = float(cap[i]); active[i] = False
        else:
            for i in range(n):
                if active[i]:
                    f[i] += lvl
            r = 0.0
    return f


def main():
    tok = sys.stdin.read().split()
    n = int(tok[0]); S = int(tok[1])
    cap = [float(x) for x in tok[2:2 + n]]

    f = waterfill([int(x) for x in cap], S)
    if HAVE_NP:
        f = np.array(f, dtype=float)
        cap_a = np.array(cap, dtype=float)

    def cur_c1(vec):
        tot = float(sum(vec))
        return 2.0 * n * peak(vec) / (tot * tot)

    rng = random.Random(20240624)
    best = cur_c1(f)
    iters = 60000
    step0 = 0.15
    for it in range(iters):
        step = step0 * (1.0 - 0.9 * it / iters)  # anneal
        i = rng.randrange(n); j = rng.randrange(n)
        if i == j:
            continue
        d = rng.random() * step
        # move d from pool j to pool i
        if f[j] - d < -1e-12 or f[i] + d > cap[i] + 1e-12:
            continue
        f[i] += d; f[j] -= d
        v = cur_c1(f)
        if v < best - 1e-12:
            best = v
        else:
            f[i] -= d; f[j] += d

    if HAVE_NP:
        out = list(f)
    else:
        out = f
    sys.stdout.write(" ".join("%.12g" % x for x in out) + "\n")


if __name__ == "__main__":
    main()
