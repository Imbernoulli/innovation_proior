# TIER: strong
# Insight: a ratio whose multiplicative order is d sweeps a coset of the unique
# subgroup H_d <= F_p^* of order d with ZERO internal waste per full lap. Grouping
# T by x -> x^d mod p (constant on cosets of H_d) turns "cover T" into a set of
# independent circular arc-cover problems, one per coset -- and trying every
# divisor d of p-1 (order selection) finds the order the target set was actually
# built from. This is the reformulation the family is named for: covering-by-
# progressions becomes quotient-group set cover once the right order is chosen.
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


def divisors_from_factor(f):
    divs = [1]
    for q, e in f.items():
        new = []
        qe = 1
        for _ in range(e + 1):
            for dd in divs:
                new.append(dd * qe)
            qe *= q
        divs = new
    return sorted(set(divs))


def primitive_root(p):
    n = p - 1
    f = factorize(n)
    pf = list(f.keys())
    g = 2
    while True:
        if all(pow(g, n // q, p) != 1 for q in pf):
            return g
        g += 1


def best_K_cost(positions, d, alpha):
    """Minimum sum(arc-lengths)+alpha*#arcs to cover `positions` (distinct points on
    a circle of circumference d) using contiguous arcs. Returns (cost, K)."""
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


def build_arcs(positions, d, K):
    """Reconstruct the actual K arcs achieving best_K_cost's optimum."""
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


def solve_for_divisor(p, alpha, T, g, n, d):
    """Cost of covering T entirely with ratio-order-d progressions, plus the data
    needed to reconstruct the chains if this divisor turns out to be the winner."""
    if d == 1:
        cost = len(T) * (1 + alpha)
        groups = [(t, [t], [0], 1) for t in T]  # leader, elems, positions, K
        return cost, ('trivial_d1', groups)

    rd = pow(g, n // d, p)
    val_to_k = {}
    val = 1
    for k in range(d):
        val_to_k[val] = k
        val = (val * rd) % p

    buckets = {}
    for x in T:
        key = pow(x, d, p)
        buckets.setdefault(key, []).append(x)

    total = 0
    groupdata = []
    for key, elems in buckets.items():
        leader = elems[0]
        inv = pow(leader, p - 2, p)
        positions = sorted(set(val_to_k[(x * inv) % p] for x in elems))
        cost, K = best_K_cost(positions, d, alpha)
        total += cost
        groupdata.append((leader, positions, K))
    return total, ('rd', rd, groupdata)


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    p = int(next(it)); alpha = int(next(it))
    m = int(next(it))
    T = [int(next(it)) for _ in range(m)]

    n = p - 1
    f = factorize(n)
    divs = divisors_from_factor(f)
    g = primitive_root(p)

    best = None  # (cost, d, payload)
    for d in divs:
        cost, payload = solve_for_divisor(p, alpha, T, g, n, d)
        if best is None or cost < best[0]:
            best = (cost, d, payload)

    _, d, payload = best
    chains = []
    if payload[0] == 'trivial_d1':
        for leader, elems, positions, K in payload[1]:
            chains.append((leader, 1, 1))
    else:
        _, rd, groupdata = payload
        for leader, positions, K in groupdata:
            arcs = build_arcs(positions, d, K)
            for start_pos, length in arcs:
                a = (leader * pow(rd, start_pos, p)) % p
                chains.append((a, rd, length))

    out = [str(len(chains))]
    for a, r, L in chains:
        out.append("%d %d %d" % (a, r, L))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
