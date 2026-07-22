"""Shared parsing/math for the fuel-blend-ladder checker (format C).
Imported ONLY by verify.py (never by the sandboxed solutions, which are self-contained)."""


def power_mean(items, a):
    """items: list of (weight>0, value>0). Weighted generalized power mean at exponent a != 0.
    z = ( sum_t w_t * y_t^a / sum_t w_t ) ^ (1/a)"""
    tw = 0.0
    s = 0.0
    for w, v in items:
        tw += w
        s += w * (v ** a)
    return (s / tw) ** (1.0 / a)


def best_price(z, corridors, p0):
    best = p0
    for (lo, hi, price) in corridors:
        if all(lo[k] <= z[k] <= hi[k] for k in range(len(z))) and price > best:
            best = price
    return best


def read_instance(path):
    toks = open(path).read().split()
    pos = [0]

    def nxt():
        v = toks[pos[0]]
        pos[0] += 1
        return v

    F = int(nxt()); M = int(nxt()); K = int(nxt()); R = int(nxt())
    a = [float(nxt()) for _ in range(K)]
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
    return dict(F=F, M=M, K=K, R=R, a=a, feed=feed, cap=cap, corridors=corridors, p0=p0)
