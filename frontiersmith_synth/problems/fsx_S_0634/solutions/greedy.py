# TIER: greedy
import sys


def nearest_stage(x, Tmin, Tmax):
    """Best (a,b) in [Tmin,Tmax]^2 approximating the single value x -- for each denominator
    b, the best numerator is round(x*b) clamped to range."""
    best = None
    bestd = None
    for b in range(Tmin, Tmax + 1):
        a = int(round(x * b))
        if a < Tmin:
            a = Tmin
        if a > Tmax:
            a = Tmax
        d = abs(a / b - x)
        if bestd is None or d < bestd:
            bestd = d
            best = (a, b)
    return best


def main():
    d = sys.stdin.read().split()
    it = iter(d)
    G = int(next(it)); Tmin = int(next(it)); Tmax = int(next(it)); K = int(next(it))
    targets = []
    for _ in range(K):
        P = int(next(it)); Q = int(next(it)); next(it)  # lambda ignored -- pure precision chase
        targets.append((P, Q))

    out = []
    for (P, Q) in targets:
        remaining = P / Q
        for i in range(G):
            left = G - i
            desired = remaining ** (1.0 / left) if remaining > 0 else 1.0
            a, b = nearest_stage(desired, Tmin, Tmax)
            out.append("%d %d" % (a, b))
            remaining = remaining / (a / b)
    print("\n".join(out))


main()
