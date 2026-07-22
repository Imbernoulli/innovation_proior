import sys, random

# gen.py <testId>  -- prints ONE pit-lane arm-scheduling instance to stdout.
#
# Format:
#   N K
#   d_1 d_2 ... d_N            (durations, 1..9)
#   s_1 s_2 ... s_N            (sectors, 0..5)
#   E
#   u_1 v_1
#   ...
#   u_E v_E                    (precedence edges, always u < v -> index order is a valid
#                               topological order)
#
# testId 1..3  : small BENIGN random instances (mild/no planted structure).
# testId 4..10 : LARGER PLANTED-TRAP instances. A "heavy" sector (sector 0) is loaded with
# many independent filler tasks PLUS one low-duration "hub" task C. C has no predecessors
# (ready at time 0, same as the fillers) but almost every task outside sector 0 depends on
# C alone. C's index is placed near the END of the sector-0 index block, well after most of
# its (precedence-free) sector-0 siblings. A forward, index-order ("ASAP") list scheduler
# therefore commits sector 0's exclusive capacity to the low-value filler siblings first
# (they appear earlier in index order and are equally "ready"), pushing C--and therefore
# almost the entire rest of the DAG, which is gated behind it--far into the schedule while
# every other arm sits idle. A scheduler that instead prioritises C (few/no precedence
# constraints of its own but a huge number of dependents) inside sector 0's queue frees the
# other 5 sectors almost immediately and keeps all arms busy, flattening the per-sector
# demand profile instead of draining it in raw index order.

SPECS_BENIGN = {
    1: (10, 3),
    2: (14, 3),
    3: (18, 3),
}

# trap tests: (F1 = #heavy sector-0 fillers placed BEFORE the hub, K = #arms,
#              ndep = #dependent tasks spread across sectors 1..5, gated behind the hub)
# Sized so that, roughly, F1 * avg_filler_dur(~7) ~= ndep * avg_dep_dur(~4) / (K-1):
# the "unlock delay" an ASAP scheduler inflicts on the hub is comparable to the time the
# other K-1 arms need to drain the dependent workload once unlocked, so a late hub (greedy)
# and an early hub (strong) land on visibly different makespans.
SPECS_TRAP = {
    4:  (8,  3, 28),
    5:  (10, 3, 35),
    6:  (12, 4, 45),
    7:  (14, 4, 55),
    8:  (16, 4, 65),
    9:  (18, 4, 75),
    10: (20, 5, 90),
}

NSEC = 6


def gen_benign(rng, N, K):
    d = [rng.randint(1, 9) for _ in range(N)]
    s = [rng.randint(0, NSEC - 1) for _ in range(N)]
    edges = []
    for v in range(2, N + 1):
        # a small number of precedence predecessors chosen from lower indices
        nreq = 0
        if rng.random() < 0.55:
            nreq = 1
        if rng.random() < 0.15:
            nreq = 2
        lo = max(1, v - 6)
        cands = list(range(lo, v))
        rng.shuffle(cands)
        for u in cands[:nreq]:
            edges.append((u, v))
    return N, K, d, s, edges


def gen_trap(rng, F1, K, ndep):
    # Sector-0 block: F1 heavy, precedence-free filler tasks (indices 1..F1), THEN the
    # hub itself at index F1+1 -- the LATEST possible position in the sector-0 block, so a
    # forward index-order scheduler drains every filler first. The hub is cheap and has NO
    # predecessors of its own (ready at time 0, tied with the fillers). Every dependent task
    # (indices F1+2 .. N) lives in sectors 1..5 and depends on the hub ALONE.
    H = F1 + 1
    hub = H
    N = H + ndep

    d = [0] * (N + 1)
    s = [0] * (N + 1)
    edges = []

    for i in range(1, F1 + 1):
        s[i] = 0
        d[i] = rng.randint(5, 9)      # heavy filler load
    s[hub] = 0
    d[hub] = rng.randint(1, 2)        # hub is cheap once it actually runs

    for k in range(ndep):
        v = H + 1 + k
        s[v] = 1 + (k % (NSEC - 1))
        d[v] = rng.randint(3, 5)
        edges.append((hub, v))

    return N, K, d[1:], s[1:], edges


def main():
    tid = int(sys.argv[1])
    rng = random.Random(7919 * tid + 13)

    if tid in SPECS_BENIGN:
        N, K = SPECS_BENIGN[tid]
        N, K, d, s, edges = gen_benign(rng, N, K)
    else:
        F1, K, ndep = SPECS_TRAP[tid]
        N, K, d, s, edges = gen_trap(rng, F1, K, ndep)

    out = []
    out.append("%d %d" % (N, K))
    out.append(" ".join(str(x) for x in d))
    out.append(" ".join(str(x) for x in s))
    out.append(str(len(edges)))
    for (u, v) in edges:
        out.append("%d %d" % (u, v))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
