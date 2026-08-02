# TIER: strong
# The insight: this is a multiple-choice knapsack in disguise. For each applicant,
# tabulate (loss, profit, tier) across all tiers and take the upper concave
# envelope in (loss, profit) space -- a tier that is worse than some cheaper
# tier in profit AND no cheaper in loss, or whose marginal profit-per-unit-loss
# is below a cheaper alternative's, can never be part of an optimal portfolio.
# Then bisect on a shadow price lambda (marginal profit per unit of loss
# budget): for each applicant pick whichever hull tier maximizes
# profit - lambda*loss, and find lambda* so the resulting portfolio spends
# almost exactly the loss cap. That rule is "assign where the marginal profit
# derivative is positive," priced against the binding portfolio constraint --
# it reads each applicant's response curve, not their raw credit score. A
# final local search mops up leftover cap budget one upgrade at a time.
import sys


def profit_loss(L, m, d_bps, u_bps, apr_bps, sev_bps):
    if m == 0:
        return 0, 0
    balance = L[m] * u_bps // 10000
    revenue = balance * apr_bps // 10000 * (10000 - d_bps) // 10000
    loss = balance * d_bps // 10000 * sev_bps // 10000
    return revenue - loss, loss


def upper_hull(points):
    """points: list of (loss, profit, tier) incl (0,0,0). Returns the
    increasing-loss upper concave envelope (strictly increasing loss, strictly
    decreasing marginal value)."""
    pts = sorted(set(points), key=lambda t: (t[0], -t[1]))
    dedup = []
    for lo, pr, m in pts:
        if dedup and dedup[-1][0] == lo:
            continue
        dedup.append((lo, pr, m))
    # Pareto filter FIRST: a point that does not strictly improve on the best
    # profit seen at any smaller-or-equal loss is dominated (a cheaper-or-equal
    # option already beats it) and must never reach the hull/lambda search --
    # without this, a later point that is both more expensive AND less
    # profitable than an earlier one (e.g. a "responsive" applicant's default
    # rate exploding at a high tier) can otherwise survive the local concavity
    # test below and get selected, silently leaving profit on the table.
    pareto = []
    best_profit = None
    for lo, pr, m in dedup:
        if best_profit is None or pr > best_profit:
            pareto.append((lo, pr, m))
            best_profit = pr
    h = []
    for lo, pr, m in pareto:
        while len(h) >= 2:
            (l1, p1, _), (l2, p2, _) = h[-2], h[-1]
            if (l2 - l1) * (pr - p1) - (p2 - p1) * (lo - l1) >= 0:
                h.pop()
            else:
                break
        h.append((lo, pr, m))
    return h


def best_for_lambda(h, lam):
    best = h[0]
    best_val = best[1] - lam * best[0]
    for lo, pr, m in h[1:]:
        v = pr - lam * lo
        if v > best_val + 1e-9:
            best_val = v
            best = (lo, pr, m)
    return best


def main():
    toks = sys.stdin.read().split()
    p = 0
    N = int(toks[p]); p += 1
    M = int(toks[p]); p += 1
    L = [0] * (M + 1)
    for m in range(1, M + 1):
        L[m] = int(toks[p]); p += 1
    apr_bps = int(toks[p]); p += 1
    sev_bps = int(toks[p]); p += 1
    cap = int(toks[p]); p += 1

    hulls = []
    for _ in range(N):
        p += 1  # score, unused by the strong solution
        d = [0] * (M + 1)
        u = [0] * (M + 1)
        for m in range(1, M + 1):
            d[m] = int(toks[p]); p += 1
            u[m] = int(toks[p]); p += 1
        pts = [(0, 0, 0)]
        for m in range(1, M + 1):
            prof, loss = profit_loss(L, m, d[m], u[m], apr_bps, sev_bps)
            pts.append((loss, prof, m))
        hulls.append(upper_hull(pts))

    lo_lam, hi_lam = 0.0, 4.0
    for _ in range(60):
        mid = (lo_lam + hi_lam) / 2.0
        tot_loss = sum(best_for_lambda(h, mid)[0] for h in hulls)
        if tot_loss > cap:
            lo_lam = mid
        else:
            hi_lam = mid
    lam = hi_lam

    choice_idx = [h.index(best_for_lambda(h, lam)) for h in hulls]
    total_loss = sum(hulls[i][choice_idx[i]][0] for i in range(N))
    remaining = cap - total_loss

    improved = True
    while improved and remaining > 0:
        improved = False
        best_i, best_gain, best_cost = -1, 0, 0
        for i in range(N):
            h = hulls[i]
            j = choice_idx[i]
            if j + 1 < len(h):
                lo1, pr1, _ = h[j]
                lo2, pr2, _ = h[j + 1]
                cost = lo2 - lo1
                gain = pr2 - pr1
                if cost <= remaining and gain > best_gain:
                    best_gain, best_cost, best_i = gain, cost, i
        if best_i >= 0:
            choice_idx[best_i] += 1
            remaining -= best_cost
            improved = True

    tiers = [hulls[i][choice_idx[i]][2] for i in range(N)]
    print(" ".join(str(x) for x in tiers))


if __name__ == "__main__":
    main()
