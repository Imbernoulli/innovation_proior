# TIER: trivial
# Reproduces the checker's own baseline: find one divisor d of N (d<=K) whose
# d-element "comb" (equally spaced, all same phase) lands on every harbor bin and
# no protected bin, then use only that one comb (ignoring the rest of the budget).
import sys


def divisors(n):
    return [i for i in range(1, n + 1) if n % i == 0]


def main():
    d = sys.stdin.read().split()
    it = iter(d)
    N = int(next(it)); K = int(next(it)); P = int(next(it))
    T = int(next(it))
    targets = [int(next(it)) for _ in range(T)]
    Q = int(next(it))
    protected = []
    for _ in range(Q):
        q = int(next(it)); next(it)
        protected.append(q)

    best = 0; bestd = None
    for dd in divisors(N):
        if dd > K:
            continue
        if any(t % dd != 0 for t in targets):
            continue
        if any(q % dd == 0 for q in protected):
            continue
        if dd > best:
            best, bestd = dd, dd

    if bestd is None:
        print(1)
        print(0, 0)
        return

    g = N // bestd
    chosen = [(m * g, 0) for m in range(bestd)]
    print(len(chosen))
    for (i, p) in chosen:
        print(i, p)


main()
