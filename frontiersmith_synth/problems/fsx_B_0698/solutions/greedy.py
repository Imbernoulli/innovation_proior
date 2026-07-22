# TIER: greedy
# The "obvious" recipe: fill tanks left-to-right, each with a 2-feedstock direct blend,
# picking the pair/split that looks best under a LINEAR (volume-weighted arithmetic mean)
# blending assumption. Never premixes (no TRANSFER moves) -- treats each tank in isolation.
# On instances where the TRUE mixing law is a power-law with exponent far from 1, the
# linear estimate mispredicts which corridor a recipe lands in, and this solution never
# discovers that an off-spec intermediate tank can be used as a tool to reach a spec no
# direct 2-feedstock blend can.
import sys


def best_price(x, corridors, p0):
    best = p0
    for (lo, hi, price) in corridors:
        if all(lo[k] <= x[k] <= hi[k] for k in range(len(x))) and price > best:
            best = price
    return best


def linear_best(cap_j, remaining, feed, K, corridors, p0):
    """Best (price_predicted, i1, i2, v) under the WRONG linear-blend assumption."""
    F = len(feed)
    best = None
    for i1 in range(F):
        if remaining[i1] <= 0:
            continue
        if remaining[i1] >= cap_j:
            z = feed[i1][1]
            pr = best_price(z, corridors, p0)
            if best is None or pr > best[0]:
                best = (pr, i1, i1, cap_j)
        for i2 in range(F):
            if i2 == i1 or remaining[i2] <= 0:
                continue
            lo_v = max(1, cap_j - remaining[i2])
            hi_v = min(cap_j - 1, remaining[i1])
            for v in range(lo_v, hi_v + 1):
                w2 = cap_j - v
                z = [(v * feed[i1][1][k] + w2 * feed[i2][1][k]) / cap_j for k in range(K)]
                pr = best_price(z, corridors, p0)
                if best is None or pr > best[0]:
                    best = (pr, i1, i2, v)
    return best


def main():
    toks = sys.stdin.read().split()
    p = 0

    def nxt():
        nonlocal p
        v = toks[p]
        p += 1
        return v

    F = int(nxt()); M = int(nxt()); K = int(nxt()); R = int(nxt())
    _a = [float(nxt()) for _ in range(K)]
    feed = []
    for _ in range(F):
        A = int(nxt())
        x = [float(nxt()) for _ in range(K)]
        feed.append((A, x))
    cap = [int(nxt()) for _ in range(M)]
    corridors = []
    for _ in range(R):
        lo = [0.0] * K
        hi = [0.0] * K
        for k in range(K):
            lo[k] = float(nxt())
            hi[k] = float(nxt())
        price = float(nxt())
        corridors.append((lo, hi, price))
    p0 = float(nxt())

    remaining = [feed[i][0] for i in range(F)]
    out = []
    for j in range(M):
        capj = cap[j]
        best = linear_best(capj, remaining, feed, K, corridors, p0)
        if best is None:
            continue
        _, i1, i2, v = best
        if i1 == i2:
            out.append("POUR %d %d %d" % (j + 1, i1 + 1, capj))
            remaining[i1] -= capj
        else:
            w2 = capj - v
            out.append("POUR %d %d %d" % (j + 1, i1 + 1, v))
            out.append("POUR %d %d %d" % (j + 1, i2 + 1, w2))
            remaining[i1] -= v
            remaining[i2] -= w2

    print("\n".join(out))


if __name__ == "__main__":
    main()
