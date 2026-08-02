import sys, random

# gen.py <testId>  -- prints ONE clock-tree buffer-insertion instance to stdout.
#
# Star clock tree: a source drives K sinks over K independent routed nets.
# Net i has a fixed wire delay w[i]; sinks with small w are "fast", sinks
# with large w are "slow". You may insert buffers (from a fixed 3-type
# library) on any net; each buffer adds delay, costs power, and shifts
# delay by a per-corner amount under each of C deterministic process
# corners (fixed numbers, not randomness). Feasibility requires (a) the
# nominal (corner-free) skew across sinks to be within a GENEROUS budget
# (you only need to close PART of the raw wire-delay gap, not all of it),
# and (b) the worst-case skew (max over corners) to be within a TIGHT
# budget sitting just above the nominal one. Objective: minimize total
# buffer power among feasible plans.
#
# Cases 1-7: benign, modest wire-delay spread (one mildly slow outlier net,
# the rest clustered together).
# Cases 8-10: trap cases, a far wider spread -- forcing a large raw gap
# that a naive "close the whole gap with the fewest buffers" strategy
# closes using the high-delay, high-variation buffer type, concentrating
# corner exposure on the lagging nets and blowing the tight worst-case-skew
# budget.

TYPES = [
    (6,  5, (-1, 0, 1, 1)),    # type 0: "safe" -- low variation
    (9,  4, (-3, 1, 2, 4)),    # type 1: "mid" -- cheap, moderate variation
    (16, 6, (-7, 2, 3, 11)),   # type 2: "turbo" -- fewest insertions, high variation
]
NTYPES = len(TYPES)
NCORNERS = 4


def full_close_with_type0(ws):
    """The checker's own reference plan: type 0 only, buffers added to every
    net below the maximum until its nominal delay reaches the max. Always
    feasible (near-zero nominal skew, low variance) but power-wasteful."""
    D0 = TYPES[0][0]
    K = len(ws)
    counts = [[0] * NTYPES for _ in range(K)]
    Tstar = max(ws)
    for i in range(K):
        gap = Tstar - ws[i]
        if gap > 0:
            counts[i][0] = -(-gap // D0)
    return counts


def eval_counts(ws, counts):
    K = len(ws)
    nom = [ws[i] + sum(counts[i][t] * TYPES[t][0] for t in range(NTYPES)) for i in range(K)]
    N = max(nom) - min(nom)
    W = 0
    for c in range(NCORNERS):
        dc = [ws[i] + sum(counts[i][t] * (TYPES[t][0] + TYPES[t][2][c]) for t in range(NTYPES))
              for i in range(K)]
        W = max(W, max(dc) - min(dc))
    B = sum(counts[i][t] * TYPES[t][1] for i in range(K) for t in range(NTYPES))
    return N, W, B


def gen_case(tid, rng):
    lo = 40
    if tid <= 7:
        K = 5 + tid  # 6..12
        width = 45  # comfortably below the trap threshold, constant across scale
    else:
        K = 5 + tid  # 13..15
        width = 200 + 60 * (tid - 8)  # R = 200..320, comfortably above it
    hi = lo + width
    jitter = max(1, width // 8)
    # one slow outlier net (the anchor) plus K-1 nets clustered near `lo`
    # with a little jitter -- deterministic extremes guarantee the exact
    # raw spread R = width every run.
    ws = [hi] + [rng.randint(lo, lo + jitter) for _ in range(K - 1)]
    rng.shuffle(ws)

    R = max(ws) - min(ws)
    nom_budget = max(32, (2 * R) // 5)         # ~0.4 R -- generous, partial closure only
                                                # (floor of 32 keeps it comfortably above
                                                # the coarsest buffer type's own rounding
                                                # granularity, so infeasibility only comes
                                                # from genuine variation exposure, not from
                                                # rounding artifacts)
    base_counts = full_close_with_type0(ws)
    _, W0, _ = eval_counts(ws, base_counts)
    slack_worst_extra = 22
    worst_budget = max(W0, nom_budget) + slack_worst_extra
    return K, ws, nom_budget, worst_budget


def main():
    tid = int(sys.argv[1])
    rng = random.Random(700000 + 97 * tid)
    K, ws, nom_budget, worst_budget = gen_case(tid, rng)

    out = []
    out.append(f"{K} {NTYPES} {NCORNERS}")
    for (D, P, Var) in TYPES:
        out.append(str(D) + " " + str(P) + " " + " ".join(str(v) for v in Var))
    out.append(f"{nom_budget} {worst_budget}")
    for w in ws:
        out.append(str(w))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
