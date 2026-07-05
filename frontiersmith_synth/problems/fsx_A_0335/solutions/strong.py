# TIER: strong
# Local-search (seeded, deterministic) coordinate descent that flattens the
# self-convolution peak below the uniform value of c1 = 2. Pure Python, O(iters * n).
import sys
import random


def peak_conv(f):
    n = len(f)
    peak = 0.0
    for k in range(2 * n - 1):
        lo = 0 if k <= n - 1 else k - (n - 1)
        hi = k if k <= n - 1 else n - 1
        acc = 0.0
        for i in range(lo, hi + 1):
            acc += f[i] * f[k - i]
        if acc > peak:
            peak = acc
    return peak


def c1(f):
    n = len(f)
    s = sum(f)
    return 2.0 * n * peak_conv(f) / (s * s)


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    rng = random.Random(1234 + 7 * n)

    best = [1.0] * n
    best_c = c1(best)
    cur = list(best)
    cur_c = best_c
    step = 0.35
    iters = 6000
    for t in range(iters):
        i = rng.randrange(n)
        old = cur[i]
        delta = rng.gauss(0.0, step)
        nv = old + delta
        if nv < 0.0:
            nv = 0.0
        cur[i] = nv
        cc = c1(cur)
        if cc <= cur_c + 1e-12:
            cur_c = cc
            if cc < best_c:
                best_c = cc
                best = list(cur)
        else:
            cur[i] = old  # reject
        if (t + 1) % 1500 == 0:
            step *= 0.6
            # occasional restart from best to escape stagnation
            cur = list(best)
            cur_c = best_c

    print(" ".join("%.10g" % v for v in best))


if __name__ == "__main__":
    main()
