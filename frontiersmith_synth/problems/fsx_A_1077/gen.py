import sys, random

# gen.py <testId>  -- prints ONE "wire the linear map" instance to stdout.
#
# HIDDEN construction (never printed -- only the resulting dense matrix A is
# shown to the solver): a TWO-LEVEL sparse butterfly factorization.
#   L1: m1 "atoms", each a +-1 combo of two distinct inputs: z = +-x[a] +-x[b].
#   L2: m2 "atoms", each a +-1 combo of two L1 atoms with DISJOINT original-
#       index support: w = +-z_p +-z_q  (so w touches 4 distinct inputs).
#   Row layer: each output row is built from 1 or 2 DISJOINT-support picks
#       from L2 (mostly) or L1 (sometimes), each with an outer random sign.
#
# Because m1, m2 are kept much smaller than the number of (row, pick)
# references, the SAME L1/L2 atom is reused by many different output rows --
# i.e. many rows share not just a partial sum but a partial-sum-of-partial-
# sums. A solver that treats each row as an independent dot product, or that
# only ever notices an accidental shared PREFIX within a single left-to-right
# pass, can see at most one level of this reuse; recovering the full benefit
# requires discovering shared structure recursively (atoms built from atoms).
#
# All "supports" are kept pairwise disjoint at each combination step, so
# every entry of the printed matrix A stays exactly in {-1, 0, 1} -- no
# coefficient stacking, no scaling instructions needed.
#
# Random (non bit-reversal) pairings and non-power-of-two n keep A far from
# any named transform (Hadamard/DFT/...) with a known-optimal addition count;
# recovering the true minimum number of additions for a general linear map is
# NP-hard, so no polynomial method can certify optimality here.

# (n, m1, m2, seed, l2_prob) per testId -- difficulty ladder. m1/m2 stay a
# small fraction of n, so reuse pressure (and hence the reward for spotting
# shared structure) grows with n; the later, larger cases are the traps.
SPECS = {
    1:  (6,   4,  3,  200,  0.75),
    2:  (7,   5,  3,  297,  0.75),
    3:  (8,   3,  2,  394,  0.75),
    4:  (9,   3,  3,  491,  0.75),
    5:  (10,  6,  2,  588,  0.75),
    6:  (12,  7,  3,  685,  0.75),   # trap: reuse pressure rises
    7:  (14,  6,  4,  782,  0.75),   # trap
    8:  (16,  9,  6,  879,  0.75),   # trap
    9:  (18,  9,  4,  976,  0.75),   # trap
    10: (20,  11, 7,  1073, 0.75),   # trap: largest n
}


def build(n, m1, m2, seed, l2_prob):
    rng = random.Random(seed)

    # ---- L1: m1 atoms, each a +-1 combo of two distinct inputs ----
    L1 = []  # list of dict: original_index -> coefficient
    for _ in range(m1):
        a, b = rng.sample(range(n), 2)
        sa, sb = rng.choice([1, -1]), rng.choice([1, -1])
        L1.append({a: sa, b: sb})

    # ---- L2: m2 atoms, each a combo of two DISJOINT-support L1 atoms ----
    L2 = []
    for _ in range(m2):
        if m1 < 2:
            break
        found = None
        for _ in range(300):
            p, q = rng.sample(range(m1), 2)
            if set(L1[p]) & set(L1[q]):
                continue
            found = (p, q)
            break
        if found is None:
            continue
        p, q = found
        s = rng.choice([1, -1])
        supp = dict(L1[p])
        for k, v in L1[q].items():
            supp[k] = supp.get(k, 0) + s * v
        supp = {k: v for k, v in supp.items() if v != 0}
        if supp:
            L2.append(supp)

    def pick_disjoint(pool, k):
        for _ in range(300):
            if k > len(pool):
                return None
            idxs = rng.sample(range(len(pool)), k)
            seen = set()
            ok = True
            for pi in idxs:
                s = set(pool[pi])
                if s & seen:
                    ok = False
                    break
                seen |= s
            if ok:
                return idxs
        return None

    A = [[0] * n for _ in range(n)]
    for i in range(n):
        row = {}
        use_l2 = (rng.random() < l2_prob) and L2
        pool = L2 if use_l2 else L1
        src = pool
        k = 2 if rng.random() < 0.35 else 1
        idxs = pick_disjoint(pool, k) if pool else None
        if idxs is None:
            idxs = pick_disjoint(L1, 1)
            src = L1
        if idxs is None:
            row[rng.randrange(n)] = 1
        else:
            for pi in idxs:
                s = rng.choice([1, -1])
                for idx, v in src[pi].items():
                    row[idx] = row.get(idx, 0) + s * v
        for idx, v in row.items():
            if v != 0:
                A[i][idx] = v

    # Guard: no all-zero row (should not happen given construction)
    for i in range(n):
        if all(v == 0 for v in A[i]):
            A[i][rng.randrange(n)] = 1

    return A


def main():
    tid = int(sys.argv[1])
    n, m1, m2, seed, l2_prob = SPECS[tid]
    A = build(n, m1, m2, seed, l2_prob)

    out = [str(n)]
    for row in A:
        out.append(" ".join(str(v) for v in row))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
