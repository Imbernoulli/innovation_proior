# TIER: trivial
# Fills each tank (in order) with a SINGLE homogeneous feedstock -- whichever one
# (with enough remaining availability) gives the best corridor price for that tank
# alone. No blending, no premixing. This mirrors the checker's own internal baseline.
import sys


def best_price(x, corridors, p0):
    best = p0
    for (lo, hi, price) in corridors:
        if all(lo[k] <= x[k] <= hi[k] for k in range(len(x))) and price > best:
            best = price
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
        best_i = -1
        best_pr = -1.0
        for i in range(F):
            if remaining[i] >= capj:
                pr = best_price(feed[i][1], corridors, p0)
                if pr > best_pr:
                    best_pr = pr
                    best_i = i
        if best_i >= 0:
            out.append("POUR %d %d %d" % (j + 1, best_i + 1, capj))
            remaining[best_i] -= capj

    print("\n".join(out))


if __name__ == "__main__":
    main()
