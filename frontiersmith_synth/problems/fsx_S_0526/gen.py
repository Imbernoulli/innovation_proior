import sys, random
from itertools import combinations

# gen.py <testId>  -- prints ONE reversible-logic-golf instance to stdout.
#
# We PLANT a boolean permutation P on {0,1}^n of the form  P = A o T  where
#   A(x) = M*x XOR b   is a GF(2)-affine bijection (M invertible over GF(2)),
#   T     = product of t Toffoli gates with DISTINCT targets drawn from the top
#           3 wires and control-pairs drawn from the low wires.
# Controls are never targets, so T is a triangular quadratic involution whose
# ANF is exactly t distinct degree-2 monomials, and T fixes every weight<=1
# vector (T(0)=0, T(e_i)=e_i). Hence P(0)=b and P(e_i)=M[:,i]^b, so the affine
# shell can be peeled directly from the table -- and the Toffoli cost then
# collapses onto the t-monomial nonlinear residue. The obvious structure-blind
# synthesis (compute-swap-uncompute over the full table's ANF) pays for the
# affine mixing redundantly across every output bit and never sees the collapse.
#
# Difficulty grows with testId (n in {6,7,8}, t in {3,4}).  Deterministic in tid.

# testId -> (n, t, seed_kick)
SPECS = {
    1:  (6, 3, 0),
    2:  (6, 3, 1),
    3:  (7, 3, 2),
    4:  (7, 3, 3),
    5:  (8, 3, 4),
    6:  (8, 4, 5),
    7:  (7, 3, 11),
    8:  (8, 4, 17),
    9:  (8, 3, 23),
    10: (8, 4, 29),
}

# Fixed cost model (quantum-style: the Toffoli / non-Clifford gate is the dear
# resource, but not so dear that the peel saturates the score cap).
COST_N = 1
COST_C = 1
COST_T = 4


def rank_gf2(rows, n):
    a = rows[:]
    r = 0
    for col in range(n):
        piv = -1
        for i in range(r, len(a)):
            if (a[i] >> col) & 1:
                piv = i; break
        if piv < 0:
            continue
        a[r], a[piv] = a[piv], a[r]
        for i in range(len(a)):
            if i != r and ((a[i] >> col) & 1):
                a[i] ^= a[r]
        r += 1
    return r


def gen_invertible(n, rng, targets):
    # random invertible M whose TARGET columns are dense (weight >= 3) so the
    # structure-blind method must recompute each residue monomial many times.
    while True:
        rows = [rng.getrandbits(n) for _ in range(n)]
        if rank_gf2(rows, n) != n:
            continue
        ok = True
        for c in targets:
            w = sum((r >> c) & 1 for r in rows)
            if w < 3:
                ok = False; break
        if ok:
            return rows


def main():
    tid = int(sys.argv[1])
    n, t, kick = SPECS[tid]
    rng = random.Random(770526 + 1000 * tid + kick)

    N = 1 << n
    tp = max(3, t)                       # size of the target block
    targets_pool = list(range(n - tp, n))
    controls_pool = list(range(n - tp))
    pairs = list(combinations(controls_pool, 2))
    assert len(pairs) >= t and len(targets_pool) >= t

    M = gen_invertible(n, rng, targets_pool)
    b = rng.getrandbits(n)

    tgts = rng.sample(targets_pool, t)
    prs = rng.sample(pairs, t)
    toffs = [(prs[i][0], prs[i][1], tgts[i]) for i in range(t)]

    def Tfun(x):
        y = x
        for (a, c, tg) in toffs:
            if ((x >> a) & 1) and ((x >> c) & 1):
                y ^= (1 << tg)
        return y

    def Afun(v):
        z = 0
        for j in range(n):
            if bin(M[j] & v).count("1") & 1:
                z |= (1 << j)
        return z ^ b

    P = [Afun(Tfun(x)) for x in range(N)]
    assert sorted(P) == list(range(N)), "not a permutation"

    a_budget = n + 2
    out = ["%d %d %d %d %d" % (n, a_budget, COST_N, COST_C, COST_T)]
    line = []
    for x in range(N):
        line.append(str(P[x]))
        if len(line) == 16:
            out.append(" ".join(line)); line = []
    if line:
        out.append(" ".join(line))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
