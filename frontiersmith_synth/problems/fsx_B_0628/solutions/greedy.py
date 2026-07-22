# TIER: greedy
# Textbook stint answer: pick the single best CONSTANT intensity and the best
# EVENLY-SPACED pit interval by brute force.  This is the first thing a strong
# coder writes -- a uniform pace -- and it ignores the stint-position structure
# of the wear externality.
import sys

def simulate(L, base, a, p, b, q, P, grid, xs, pits):
    wear = 0.0; total = 0.0
    for i in range(L):
        if pits[i]:
            total += P; wear = 0.0
        x = grid[xs[i]]
        total += base + a * (wear ** p) + b / x
        wear += x ** q
    return total

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    L = int(next(it)); k = int(next(it))
    base = float(next(it)); a = float(next(it)); p = float(next(it))
    b = float(next(it)); q = float(next(it)); P = float(next(it))
    grid = [float(next(it)) for _ in range(k)]

    best = None
    for j in range(k):
        for t in [0] + list(range(2, L + 1)):
            xs = [j] * L
            pits = [0] * L
            if t:
                for i in range(t, L, t):
                    pits[i] = 1
            c = simulate(L, base, a, p, b, q, P, grid, xs, pits)
            if best is None or c < best[0]:
                best = (c, xs, pits)
    _, xs, pits = best
    out = ["%d %d" % (xs[i], pits[i]) for i in range(L)]
    sys.stdout.write("\n".join(out) + "\n")

main()
