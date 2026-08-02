import sys, random

# ---- transaction-isolation-choose instance generator ----------------------
# N transactions, each with a read-set and a write-set drawn from a shared
# key space, plus a throughput weight. Two STATIC conflict structures are
# planted directly from the read/write sets (no runtime scheduling needed):
#
#   rw-hazard edge i->j   iff  R_i and W_j share a key   (i "anti-depends" on j)
#   ww-pair    {i,j}      iff  W_i and W_j share a key   (lost-update risk)
#
# Isolation levels: 0=READ COMMITTED, 1=SNAPSHOT, 2=SERIALIZABLE.
# A ww-pair is dangerous only if BOTH sides sit at level 0.
# An rw-edge is dangerous only if BOTH endpoints sit at level <= 1; a
# directed CYCLE built entirely from such "exposed" rw-edges is the anomaly
# (classic write-skew). Breaking a cycle needs only ONE member promoted all
# the way to level 2 -- not everyone.
#
# TRAP testIds (4,6,8,9,10) plant rw-cycles (length 2, 3 or 4) where every
# cycle member has conflict-degree exactly 1 or 2 (conflicts only with its
# cycle neighbours). A degree-threshold recipe ("few conflicts -> snapshot
# is enough") buckets every one of them at level 1 and never notices the
# cycle survives -- it must actually walk the directed graph.


def make_alloc():
    ctr = [0]
    def alloc():
        ctr[0] += 1
        return ctr[0] - 1
    return alloc, ctr


def build_cycle(n, rng, alloc, wlo, whi):
    """n transactions forming one directed rw-cycle 0->1->...->(n-1)->0.
    Each transaction i: R_i=[K_i], W_i=[K_{i-1 mod n}]."""
    keys = [alloc() for _ in range(n)]
    txns = []
    for i in range(n):
        w = rng.randint(wlo, whi)
        R = [keys[i]]
        W = [keys[(i - 1) % n]]
        txns.append((w, R, W))
    return txns


def build_chain(n, rng, alloc, wlo, whi):
    """n transactions forming an ACYCLIC rw-path 0->1->...->(n-1) (no
    wraparound edge -- never violates, regardless of chosen levels)."""
    keys = [alloc() for _ in range(n - 1)]
    txns = []
    for i in range(n):
        w = rng.randint(wlo, whi)
        R = [keys[i]] if i < n - 1 else []
        W = [keys[i - 1]] if i >= 1 else []
        txns.append((w, R, W))
    return txns


def build_ww_group(m, rng, alloc, wlo, whi):
    """m transactions all writing one shared key -> complete ww-clique.
    No reads, so no accidental rw edges."""
    z = alloc()
    txns = []
    for _ in range(m):
        w = rng.randint(wlo, whi)
        txns.append((w, [], [z]))
    return txns


def build_fillers(count, rng, alloc, wlo, whi):
    """Fully independent transactions: each gets a fresh private read key
    and a fresh private write key, touched by nobody else."""
    txns = []
    for _ in range(count):
        w = rng.randint(wlo, whi)
        rk = alloc()
        wk = alloc()
        txns.append((w, [rk], [wk]))
    return txns


def build(t, rng, alloc):
    WLO, WHI = 5, 25
    if t == 1:
        return build_fillers(4, rng, alloc, WLO, WHI)
    if t == 2:
        return build_ww_group(2, rng, alloc, WLO, WHI) + build_fillers(3, rng, alloc, WLO, WHI)
    if t == 3:
        return build_chain(3, rng, alloc, WLO, WHI) + build_fillers(3, rng, alloc, WLO, WHI)
    if t == 4:
        return build_cycle(2, rng, alloc, WLO, WHI) + build_fillers(4, rng, alloc, WLO, WHI)
    if t == 5:
        return build_ww_group(4, rng, alloc, WLO, WHI) + build_fillers(3, rng, alloc, WLO, WHI)
    if t == 6:
        return build_cycle(3, rng, alloc, WLO, WHI) + build_fillers(5, rng, alloc, WLO, WHI)
    if t == 7:
        return (build_chain(4, rng, alloc, WLO, WHI)
                + build_ww_group(2, rng, alloc, WLO, WHI)
                + build_fillers(3, rng, alloc, WLO, WHI))
    if t == 8:
        return (build_cycle(2, rng, alloc, WLO, WHI)
                + build_cycle(2, rng, alloc, WLO, WHI)
                + build_fillers(6, rng, alloc, WLO, WHI))
    if t == 9:
        return (build_cycle(4, rng, alloc, WLO, WHI)
                + build_ww_group(2, rng, alloc, WLO, WHI)
                + build_fillers(6, rng, alloc, WLO, WHI))
    if t == 10:
        return (build_cycle(3, rng, alloc, WLO, WHI)
                + build_cycle(2, rng, alloc, WLO, WHI)
                + build_ww_group(3, rng, alloc, WLO, WHI)
                + build_fillers(6, rng, alloc, WLO, WHI))
    raise ValueError("testId out of range")


def main():
    t = int(sys.argv[1])
    rng = random.Random(20260726 + 97 * t)
    alloc, ctr = make_alloc()

    txns = build(t, rng, alloc)
    N = len(txns)
    K = ctr[0]

    lines = [f"{N} {K}"]
    for (w, R, W) in txns:
        parts = [str(w), str(len(R))] + [str(k) for k in R] + [str(len(W))] + [str(k) for k in W]
        lines.append(" ".join(parts))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
