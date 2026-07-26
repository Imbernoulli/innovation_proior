# TIER: strong
"""Hybrid coset-skeleton + interval-shim construction.

Insight: label trees by their GRAFT index (discrete log base a primitive
root g), not by natural position. A plain "index mod k" coset is perfectly
graft-balanced for every divisor d that does NOT divide k (orthogonality:
summing a nontrivial residue class evenly over a full coset cancels), but is
CATASTROPHIC for any tested d that DOES divide k (the whole coset then
shares one fixed residue mod d -- total concentration, not spread).

Fix: let D = lcm of every tested graft divisor that divides k (the
"conflicting" orders). Use a finer skeleton modulus M = k*D. A tree's final
plot is (graft_index mod M) // D -- i.e. group D consecutive log-index
residues (an "interval shim" in log-space) into one plot instead of a single
residue. Within a plot, the log index now sweeps through ALL D residues mod
D exactly once per M-cycle, which by construction also covers every residue
mod every conflicting divisor exactly once -- restoring perfect graft
balance for the divisors that broke the plain coset. Because plots are still
built from GRAFT-index ranges (not natural-value ranges), row/window balance
stays good too, by the same pseudorandom-spread argument that made the plain
coset good on rows: the discrete-log map scatters any log-contiguous set
across natural position space.

Every graft divisor NOT dividing k needs no correction (plain coset already
handles it); folding it into D would only make the skeleton coarser, so D is
built from conflicting divisors only.
"""
import sys
from math import gcd


def factorize(n):
    f = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            f[d] = f.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        f[n] = f.get(n, 0) + 1
    return f


def find_primitive_root(p):
    n = p - 1
    fac = factorize(n)
    for g in range(2, p):
        ok = True
        for q in fac:
            if pow(g, n // q, p) == 1:
                ok = False
                break
        if ok:
            return g
    return None


def build_idx(p, g):
    n = p - 1
    idx_arr = [0] * p
    val = 1
    for i in range(n):
        idx_arr[val] = i
        val = (val * g) % p
    return idx_arr


def lcm(a, b):
    return a * b // gcd(a, b)


def main():
    data = sys.stdin.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        v = data[pos]
        pos += 1
        return v

    p = int(nxt())
    k = int(nxt())
    sizes = [int(nxt()) for _ in range(k)]
    m_row = int(nxt())
    for _ in range(m_row):
        nxt(); nxt()  # windows unused by the skeleton itself
    m_graft = int(nxt())
    grafts = [int(nxt()) for _ in range(m_graft)]

    n = p - 1
    bs = sizes[0] if sizes and all(s == sizes[0] for s in sizes) else n // k

    conflicting = [d for d in grafts if k % d == 0]
    D = 1
    for d in conflicting:
        D = lcm(D, d)
    M = k * D

    g = find_primitive_root(p)
    idx_arr = build_idx(p, g)

    if n % M != 0:
        # Skeleton modulus does not evenly tile this instance (shouldn't
        # happen on generated instances) -- fall back to a safe, still-valid
        # partition that at least meets the prescribed sizes exactly.
        out = []
        for i in range(1, k + 1):
            out.extend([str(i)] * sizes[i - 1])
        sys.stdout.write(" ".join(out) + "\n")
        return

    labels = [0] * (n + 1)
    for x in range(1, n + 1):
        i = idx_arr[x]
        plot = (i % M) // D + 1
        labels[x] = plot

    out = [str(labels[x]) for x in range(1, n + 1)]
    sys.stdout.write(" ".join(out) + "\n")


if __name__ == "__main__":
    main()
