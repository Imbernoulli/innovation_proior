# TIER: strong
"""
Two decisions drive the objective, and each has a genuine near-optimal answer once you stop
treating them as "always gate everything, always split to the max":

1. THRESHOLD, given a domain's SIZE s (its leakage rate is L*s per ON step -- a domain is a
   shared power rail, so bundling more blocks onto it makes every ON step, active or idle,
   proportionally pricier): gating an idle run of length Lr replaces L*s*Lr of leakage with a
   flat W. That's a strict win iff Lr*L*s > W, i.e. iff Lr >= floor(W/(L*s)) + 1 =: theta*(s).
   BIGGER domains have a LOWER breakeven length -- a rail carrying more blocks should gate
   more eagerly, not less -- which a size-blind threshold (one flat number for every domain)
   gets wrong in both directions.

2. GROUPING (which blocks share a domain): merging shrinks a domain's idle windows (OR only
   adds 1s) but also shrinks the NUMBER of separate rails paying their own wakeup churn, and
   dilutes the multiplied per-step leakage-rate cost of frequently-toggling blocks across
   fewer, better-amortized transitions. Whether a given merge is a net win or a net loss is
   therefore data-dependent, not a fixed rule, and it keeps changing as components grow (two
   blocks that look cheap to fuse in isolation can become a bad idea once one of them is
   already part of a larger correlated group) -- so this is a proper AGGLOMERATIVE clustering:
   maintain the true net energy delta for every pair of CURRENT components, always take the
   cheapest merge, and re-price only the merged component's row against everyone else after
   each step (so the decision that fused the correlated cluster together doesn't go stale and
   get reused to justify folding unrelated blocks into it later). Free (delta<=0) merges are
   always taken; costly ones are taken only while the D-domain cap still forces it, cheapest
   first.
"""
import sys
from itertools import groupby


def domain_energy(rows_per_trace, blocks, L, W):
    """Energy of one domain (a fixed block set), at ITS OWN size-optimal threshold, across
    all K traces. Returns 0 for an empty block set (no rail, no leakage)."""
    s = len(blocks)
    if s == 0:
        return 0
    rate = L * s
    theta = W // rate + 1
    total = 0
    for rows in rows_per_trace:
        if s == 1:
            demand = rows[blocks[0]]
        else:
            combo = 0
            for b in blocks:
                combo |= int(rows[b], 2)
            T = len(rows[blocks[0]])
            demand = format(combo, "0{}b".format(T))
        runs = [(k, len(list(g))) for k, g in groupby(demand)]
        n_runs = len(runs)
        for ridx, (ch, length) in enumerate(runs):
            if ch == "1":
                total += rate * length
            else:
                is_last = (ridx == n_runs - 1)
                if length >= theta:
                    if not is_last:
                        total += W
                else:
                    total += rate * length
    return total


def agglomerate(rows_per_trace, N, L, W, target):
    """Incremental agglomerative clustering (correct, not a one-shot static-cost Kruskal):
    start from N singletons, repeatedly merge the CURRENT pair with the lowest net energy
    delta -- taking every free (delta<=0) merge, and additional costly merges only while more
    than `target` components remain -- re-pricing only the freshly merged component against
    every survivor after each step. Returns a list of block-index lists (final groups)."""
    comp_id = list(range(N))          # block -> current component id
    members = {c: [c] for c in range(N)}
    energy = {c: domain_energy(rows_per_trace, [c], L, W) for c in range(N)}
    alive = set(range(N))

    # pairwise delta cache: (a,b) with a<b -> delta
    delta = {}
    for a in alive:
        for b in alive:
            if a < b:
                e = domain_energy(rows_per_trace, members[a] + members[b], L, W)
                delta[(a, b)] = e - energy[a] - energy[b]

    while True:
        if not delta:
            break
        (a, b), best = min(delta.items(), key=lambda kv: kv[1])
        if len(alive) <= target and best > 0:
            break  # cap already satisfied, nothing left is a free win
        # merge b into a
        new_members = members[a] + members[b]
        new_energy = domain_energy(rows_per_trace, new_members, L, W)
        members[a] = new_members
        energy[a] = new_energy
        for blk in members[b]:
            comp_id[blk] = a
        del members[b]
        del energy[b]
        alive.discard(b)
        # drop every stale pair touching a or b, then re-price a against every survivor
        for key in list(delta.keys()):
            if a in key or b in key:
                del delta[key]
        for other in alive:
            if other == a:
                continue
            lo, hi = (a, other) if a < other else (other, a)
            e = domain_energy(rows_per_trace, members[a] + members[other], L, W)
            delta[(lo, hi)] = e - energy[a] - energy[other]

    return [members[c] for c in alive]


def main():
    data = sys.stdin.read().split()
    p = iter(data)

    def nx():
        return next(p)

    N = int(nx())
    D = int(nx())
    K = int(nx())
    T = int(nx())
    L = int(nx())
    W = int(nx())
    rows_per_trace = []
    for _k in range(K):
        rows_per_trace.append([nx() for _ in range(N)])

    target = min(N, D)
    groups_list = agglomerate(rows_per_trace, N, L, W, target)

    dom = [0] * N
    for did, blocks in enumerate(groups_list, start=1):
        for b in blocks:
            dom[b] = did
    Du = len(groups_list)
    theta = []
    for blocks in groups_list:
        s = len(blocks)
        rate = L * s
        theta.append(W // rate + 1)

    out = [str(Du), " ".join(map(str, dom)), " ".join(map(str, theta))]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
