# TIER: strong
import sys
from bisect import bisect_left


def build_single(Tmin, Tmax):
    opts = []
    for a in range(Tmin, Tmax + 1):
        for b in range(Tmin, Tmax + 1):
            opts.append((a, b, a / b, a + b))
    return opts


def build_pairs(opts):
    """All achievable two-stage products, sorted by value -- this is the joint search
    table: instead of rounding each stage independently, we can binary-search this table
    for the two-stage combination that best complements any choice of the remaining stage."""
    pairs = []
    for (a1, b1, v1, c1) in opts:
        for (a2, b2, v2, c2) in opts:
            pairs.append((v1 * v2, c1 + c2, a1, b1, a2, b2))
    pairs.sort(key=lambda t: t[0])
    return pairs


def solve_target(P, Q, lam, G, Tmin, Tmax, opts, pairs, pair_vals, norm):
    target = P / Q
    best = None

    if G == 3:
        for (a3, b3, v3, c3) in opts:
            need = target / v3
            pos = bisect_left(pair_vals, need)
            lo = max(0, pos - 6)
            hi = min(len(pairs), pos + 6)
            for j in range(lo, hi):
                v12, c12, a1, b1, a2, b2 = pairs[j]
                V = v12 * v3
                relerr = abs(V - target) / target
                cost = c12 + c3
                contrib = relerr + lam * (cost / norm)
                if best is None or contrib < best[0]:
                    best = (contrib, a1, b1, a2, b2, a3, b3)
        _, a1, b1, a2, b2, a3, b3 = best
        return [(a1, b1), (a2, b2), (a3, b3)]

    if G == 2:
        for (a1, b1, v1, c1) in opts:
            need = target / v1
            pos = bisect_left(pair_vals, need)  # here pair_vals is single-stage sorted values
            lo = max(0, pos - 4)
            hi = min(len(pairs), pos + 4)
            for j in range(lo, hi):
                a2, b2, v2, c2 = pairs[j]
                V = v1 * v2
                relerr = abs(V - target) / target
                cost = c1 + c2
                contrib = relerr + lam * (cost / norm)
                if best is None or contrib < best[0]:
                    best = (contrib, a1, b1, a2, b2)
        _, a1, b1, a2, b2 = best
        return [(a1, b1), (a2, b2)]

    # generic fallback for other G (not exercised by this problem's own generator):
    # sequential nearest-root rounding, but still cost-aware at the last stage.
    remaining = target
    chosen = []
    for i in range(G):
        left = G - i
        desired = remaining ** (1.0 / left) if remaining > 0 else 1.0
        bd = None
        bo = None
        for (a, b, v, c) in opts:
            d = abs(v - desired) + lam * (c / norm)
            if bd is None or d < bd:
                bd = d
                bo = (a, b, v)
        chosen.append((bo[0], bo[1]))
        remaining = remaining / bo[2]
    return chosen


def main():
    d = sys.stdin.buffer.read().split()
    it = iter(d)
    G = int(next(it)); Tmin = int(next(it)); Tmax = int(next(it)); K = int(next(it))
    targets = []
    for _ in range(K):
        P = int(next(it)); Q = int(next(it)); lam = float(next(it))
        targets.append((P, Q, lam))

    opts = build_single(Tmin, Tmax)
    norm = float(G * (Tmin + Tmax))

    if G == 3:
        pairs = build_pairs(opts)
        pair_vals = [p[0] for p in pairs]
    elif G == 2:
        pairs = sorted(opts, key=lambda t: t[2])
        pair_vals = [p[2] for p in pairs]
    else:
        pairs = None
        pair_vals = None

    out = []
    for (P, Q, lam) in targets:
        for (a, b) in solve_target(P, Q, lam, G, Tmin, Tmax, opts, pairs, pair_vals, norm):
            out.append("%d %d" % (a, b))
    print("\n".join(out))


main()
