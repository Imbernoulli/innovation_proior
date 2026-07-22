# TIER: greedy
# The "obvious" recipe: a primitive root reaches every village eventually, so just
# use ONE fixed generator (order p-1, no search over other multiplicative orders)
# and greedily arc-cover T's positions along that single cycle. This still spends
# effort trimming unneeded stretches (skipping the biggest gaps), but never asks
# whether a SMALLER order would make some progression sweep a whole coset for
# free -- so it pays close to full coupon-collector-style waste on any cluster
# that was actually built from a smaller subgroup.
import sys


def factorize(n):
    f = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            f[d] = f.get(d, 0) + 1
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        f[n] = f.get(n, 0) + 1
    return f


def primitive_root(p):
    n = p - 1
    f = factorize(n)
    pf = list(f.keys())
    g = 2
    while True:
        if all(pow(g, n // q, p) != 1 for q in pf):
            return g
        g += 1


def build_arcs(positions, d, K):
    m = len(positions)
    if m == 1:
        return [(positions[0], 1)]
    gaps = []
    for i in range(m):
        a = positions[i]
        b = positions[(i + 1) % m]
        if i < m - 1:
            g = b - a - 1
        else:
            g = (d - a - 1) + b
        gaps.append(g)
    order = sorted(range(m), key=lambda i: (-gaps[i], i))
    cuts = sorted(order[:K])
    ncuts = len(cuts)
    arcs = []
    for t in range(ncuts):
        start_idx = (cuts[t] + 1) % m
        end_idx = cuts[(t + 1) % ncuts]
        start_pos = positions[start_idx]
        end_pos = positions[end_idx]
        length = ((end_pos - start_pos) % d) + 1
        arcs.append((start_pos, length))
    return arcs


def best_K_cost(positions, d, alpha):
    m = len(positions)
    if m == 1:
        return 1 + alpha, 1
    gaps = []
    for i in range(m):
        a = positions[i]
        b = positions[(i + 1) % m]
        if i < m - 1:
            g = b - a - 1
        else:
            g = (d - a - 1) + b
        gaps.append(g)
    gaps.sort(reverse=True)
    prefix = [0]
    for gv in gaps:
        prefix.append(prefix[-1] + gv)
    best_cost, best_K = None, 1
    for K in range(1, m + 1):
        length = d - prefix[K]
        cost = length + alpha * K
        if best_cost is None or cost < best_cost:
            best_cost, best_K = cost, K
    return best_cost, best_K


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    p = int(next(it)); alpha = int(next(it))
    m = int(next(it))
    T = [int(next(it)) for _ in range(m)]

    n = p - 1
    g = primitive_root(p)  # the "obvious" ratio: order n, reaches everywhere

    # Fermat: x^n == 1 for every x in F_p^*, so a fixed generator never partitions
    # T by any subgroup structure -- every village lands in the SAME single bucket.
    chains = []
    if len(T) == 1:
        chains.append((T[0], 1, 1))
    else:
        pos_list = sorted(set(_log(x, g, p, n) for x in T))
        cost, K = best_K_cost(pos_list, n, alpha)
        arcs = build_arcs(pos_list, n, K)
        for start_pos, length in arcs:
            a = pow(g, start_pos, p)
            chains.append((a, g, length))

    out = [str(len(chains))]
    for a, r, L in chains:
        out.append("%d %d %d" % (a, r, L))
    sys.stdout.write("\n".join(out) + "\n")


def _log(x, g, p, n):
    """Discrete log of x base g mod p (order n = p-1): build once, reuse via cache."""
    if not hasattr(_log, "_table"):
        _log._table = {}
        _log._key = None
    if _log._key != (g, p):
        table = {}
        val = 1
        for k in range(n):
            table[val] = k
            val = (val * g) % p
        _log._table = table
        _log._key = (g, p)
    return _log._table[x]


if __name__ == "__main__":
    main()
