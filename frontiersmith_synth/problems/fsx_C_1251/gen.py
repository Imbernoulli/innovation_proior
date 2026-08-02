#!/usr/bin/env python3
"""gen.py <testId> -- instance generator for instruction-set-select (fsx_C_1251).

Builds a set of candidate custom instructions (subgraph fusions) with an area cost
and an encoding-space cost of exactly one slot each, plus one or more application
code streams in which each candidate has a list of match occurrences (positions it
could replace with a fused op). Occurrences are constructed so that within a
"cluster" several distinct candidate flavors occur at the EXACT SAME code interval
(mutually exclusive alternative fusions of one code region -- the overlap trap),
while a long tail of candidates occur at disjoint, non-overlapping positions
(genuinely additive coverage). Deterministic: all randomness is seeded from testId.
"""
import random
import sys


def params(t):
    # (M apps, clusters=[(flavors, instances), ...], n_tail, tail_freq_range, tail_size_range)
    table = {
        1: dict(M=1, clusters=[], tails=9, tfreq=(1, 3), tsize=(2, 4)),
        2: dict(M=1, clusters=[(3, 7)], tails=11, tfreq=(1, 3), tsize=(2, 4)),
        3: dict(M=2, clusters=[(3, 8)], tails=14, tfreq=(1, 4), tsize=(2, 5)),
        4: dict(M=1, clusters=[(4, 10), (3, 9)], tails=16, tfreq=(1, 3), tsize=(2, 5)),
        5: dict(M=1, clusters=[(4, 12), (4, 11), (3, 9)], tails=19, tfreq=(1, 3), tsize=(2, 5)),
        6: dict(M=2, clusters=[(5, 12), (4, 11), (3, 10)], tails=21, tfreq=(1, 4), tsize=(2, 5)),
        7: dict(M=1, clusters=[(4, 13), (4, 12), (4, 11), (3, 10)], tails=25, tfreq=(1, 4), tsize=(2, 6)),
        8: dict(M=2, clusters=[(5, 14), (4, 13), (4, 12), (3, 11)], tails=29, tfreq=(1, 4), tsize=(2, 6)),
        9: dict(M=3, clusters=[(5, 15), (5, 14), (4, 13), (4, 12), (3, 11)], tails=31, tfreq=(1, 4), tsize=(2, 6)),
        10: dict(M=3, clusters=[(5, 16), (5, 15), (4, 14), (4, 13), (3, 12)], tails=36, tfreq=(1, 4), tsize=(2, 6)),
    }
    return table[max(1, min(10, t))]


def main():
    t = int(sys.argv[1])
    p = params(t)
    rng = random.Random(20260 + 97 * t)

    # ---- build a shuffled unit ORDER first, so candidate ids are interleaved
    #      between cluster flavors and tail candidates (id order must carry no
    #      signal about group membership -- that's what keeps the "trivial"
    #      id-order baseline genuinely naive rather than accidentally frequency-
    #      correlated) ------------------------------------------------------
    specs = [("cluster", nf, ninst) for (nf, ninst) in p["clusters"]]
    specs += [("tail", None, None) for _ in range(p["tails"])]
    rng.shuffle(specs)

    cand_area, cand_size, cand_cost = [], [], []
    cluster_groups = []   # each: (list of cand_ids sharing identical span, span, num_instances)
    tail_units = []       # each: (cand_id, size, freq)

    for kind, nf, ninst in specs:
        if kind == "cluster":
            span = nf + rng.randint(1, 3)
            costs = rng.sample(range(1, span), nf)  # distinct fusion costs < span
            ids = []
            for c in costs:
                cid = len(cand_area)
                cand_area.append(rng.randint(3, 7))
                cand_size.append(span)
                cand_cost.append(c)
                ids.append(cid)
            cluster_groups.append((ids, span, ninst))
        else:
            sz = rng.randint(*p["tsize"])
            cost = rng.randint(1, sz - 1)
            cid = len(cand_area)
            cand_area.append(rng.randint(1, 3))
            cand_size.append(sz)
            cand_cost.append(cost)
            freq = rng.randint(*p["tfreq"])
            tail_units.append((cid, sz, freq))

    C = len(cand_area)

    # ---- lay out placement units (independently reshuffled) across the M
    #      applications --------------------------------------------------
    M = p["M"]
    app_cursor = [0] * M
    app_occ = [[] for _ in range(M)]  # list of (cand_id, start)

    units = [("cluster", cg) for cg in cluster_groups] + [("tail", tu) for tu in tail_units]
    rng.shuffle(units)

    for kind, payload in units:
        m = rng.randrange(M)
        if kind == "cluster":
            ids, span, ninst = payload
            for _ in range(ninst):
                gap = rng.randint(0, 2)
                app_cursor[m] += gap
                start = app_cursor[m]
                for cid in ids:
                    app_occ[m].append((cid, start))
                app_cursor[m] += span
        else:
            cid, sz, freq = payload
            for _ in range(freq):
                gap = rng.randint(0, 2)
                app_cursor[m] += gap
                start = app_cursor[m]
                app_occ[m].append((cid, start))
                app_cursor[m] += sz

    # pad each app with a little trailing plain code so L_m isn't a razor edge
    L = [app_cursor[m] + rng.randint(1, 4) for m in range(M)]

    # ---- budgets ------------------------------------------------------------
    n_clusters = len(p["clusters"])
    n_tail = p["tails"]
    K = n_clusters + max(3, round(0.28 * n_tail))
    total_area = sum(cand_area)
    A = max(cand_area) + round(0.5 * total_area)
    # ensure genuine tradeoff: budgets must not be able to buy everything
    K = min(K, C - 1)
    if K < 1:
        K = 1

    # ---- emit ---------------------------------------------------------------
    out = []
    out.append(f"{K} {A}")
    out.append(f"{C}")
    for c in range(C):
        out.append(f"{cand_area[c]} {cand_size[c]} {cand_cost[c]}")
    out.append(f"{M}")
    for m in range(M):
        out.append(f"{L[m]} {len(app_occ[m])}")
        for cid, start in app_occ[m]:
            out.append(f"{cid} {start}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
