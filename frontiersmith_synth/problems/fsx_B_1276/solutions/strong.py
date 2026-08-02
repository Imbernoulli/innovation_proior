# TIER: strong
import sys, math


def nearest_rank(sorted_vals_weights, p):
    W = sum(w for _, w in sorted_vals_weights)
    if W <= 0:
        return None
    pos = math.ceil(p * W)
    pos = max(1, min(W, pos))
    c = 0
    for v, w in sorted_vals_weights:
        c += w
        if c >= pos:
            return v
    return sorted_vals_weights[-1][0]


def main():
    d = sys.stdin.read().split()
    it = iter(d)
    N = int(next(it))
    f0 = int(next(it)); a0 = int(next(it)); r0 = int(next(it))
    REV = int(next(it)); BUDGET = int(next(it)); MIN_COMPS = int(next(it))
    comps = []
    for _ in range(N):
        margin = int(next(it)); f = int(next(it)); a = int(next(it))
        r = int(next(it)); doc_cost = int(next(it))
        comps.append((margin, f, a, r, doc_cost))

    # insight: hedge against the UNKNOWN per-axis audit weighting by preferring candidates
    # that are close on EVERY axis (worst-axis / Chebyshev distance), not just on average --
    # a candidate that is merely close in L1 can still be extreme on one axis that some
    # posture weights heavily. Submitting is never free (budget), so breadth is spent
    # deliberately on the most-robust candidates first, not maximized blindly.
    scored = []
    for idx in range(N):
        margin, f, a, r, doc_cost = comps[idx]
        cheb = max(abs(f - f0), abs(a - a0), abs(r - r0))
        scored.append((cheb, idx))
    scored.sort()
    order_idx = [idx for cheb, idx in scored]

    cap = max(MIN_COMPS + 2, min(N, MIN_COMPS + 9))
    chosen = []
    spent = 0
    for idx in order_idx:
        if len(chosen) >= cap:
            break
        cost = comps[idx][4]
        if spent + cost <= BUDGET:
            chosen.append(idx + 1)
            spent += cost
    if len(chosen) < MIN_COMPS:
        chosen = [idx + 1 for idx in order_idx[:MIN_COMPS]]
        spent = sum(comps[i1 - 1][4] for i1 in chosen)
    remaining = BUDGET - spent

    # spend any LEFTOVER budget defending the weakest-fit (highest worst-axis deviation)
    # members of the chosen set -- that is where scrutiny is most likely to bite
    order = sorted(chosen, key=lambda i1: -max(
        abs(comps[i1 - 1][1] - f0), abs(comps[i1 - 1][2] - a0), abs(comps[i1 - 1][3] - r0)))
    doc_depth = {i1: 0 for i1 in chosen}
    for depth in (1, 2, 3):
        for i1 in order:
            cost = comps[i1 - 1][4]
            if doc_depth[i1] == depth - 1 and remaining >= cost:
                doc_depth[i1] = depth
                remaining -= cost

    # defensible position: a percentile of the margins of the balanced (Chebyshev-close)
    # HALF of the full visible universe -- a broad, representative sample (independent of
    # what the budget can afford to formally submit) that tracks the true range under most
    # unknown per-axis weightings, instead of a narrow budget- or margin-biased subset
    half_cut = scored[min(len(scored) - 1, max(MIN_COMPS + 3, N // 2))][0]
    universe_margins = sorted(comps[idx][0] for cheb, idx in scored if cheb <= half_cut)
    if not universe_margins:
        universe_margins = sorted(comps[i1 - 1][0] for i1 in chosen)
    M = nearest_rank([(v, 1) for v in universe_margins], 0.75)

    print(len(chosen))
    for i1 in chosen:
        print(i1, doc_depth[i1])
    print(M)


main()
