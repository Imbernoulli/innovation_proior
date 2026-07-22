# TIER: greedy
# The obvious recipe: at each stage pour into the currently most-deficient cell
# just enough to lift IT to target.  It matches the target cell-by-cell, ignoring
# that settling spreads grains downhill and that overshoot is charged at EVERY
# later stage -> it over-pours early and pays a large integrated penalty.
import sys


def settle(h, S):
    N = len(h)
    while True:
        stable = True
        for i in range(N - 1):
            d = h[i] - h[i + 1]
            if d > S:
                m = (d - S + 1) // 2
                h[i] -= m; h[i + 1] += m; stable = False
            elif -d > S:
                m = (-d - S + 1) // 2
                h[i + 1] -= m; h[i] += m; stable = False
        if stable:
            return


def main():
    d = sys.stdin.read().split()
    it = iter(d)
    N = int(next(it)); K = int(next(it)); S = int(next(it))
    L = int(next(it)); G = int(next(it))
    t = [int(next(it)) for _ in range(N)]

    h = [0] * N
    out = []
    for _ in range(K):
        # most-deficient cell
        c = max(range(N), key=lambda i: t[i] - h[i])
        deficit = t[c] - h[c]
        g = deficit if deficit > 0 else 0
        if g > G:
            g = G
        out.append("%d %d" % (c, g))
        h[c] += g
        settle(h, S)
    print("\n".join(out))


main()
