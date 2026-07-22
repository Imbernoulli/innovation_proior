# TIER: strong
import sys


def sweep_optimal(cands):
    """cands: list of (P_max, expected_payout). Return (bestP, bestProfit)
    for the exact profit-maximizing flat price over this candidate list."""
    cands = sorted(cands, key=lambda t: -t[0])
    best_p, best_profit = 0.0, 0.0
    prefix = 0.0
    for k, (pm, payout) in enumerate(cands, 1):
        prefix += payout
        profit = pm * k - prefix
        if profit > best_profit:
            best_p, best_profit = pm, profit
    return best_p, best_profit


def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    agents = []
    for _ in range(n):
        p = float(data[idx]); a = float(data[idx + 1]); L = float(data[idx + 2]); u = float(data[idx + 3])
        idx += 4
        agents.append((p, a, L, u))

    if n == 0:
        print(0)
        return

    # INSIGHT 1 (participation-exit): the binding question is not how to price
    # everyone, it is who is even profitable to serve. At full coverage the
    # residual-risk penalty vanishes, so the maximum extractable margin from
    # agent i collapses to a clean number: ceiling_i = -u_i - p_i*L_i. Filter
    # to the servable set BEFORE doing anything else -- this is exactly what
    # greedy never computes (it never reads u_i at all).
    servable = [(p, a, L, u) for (p, a, L, u) in agents if (-u - p * L) > 1e-9]
    if not servable:
        print(0)
        return

    # INSIGHT 2 (adverse-selection-replay / contract-menu-screening): among the
    # servable set, split by the largest gap in true risk p_i -- this isolates
    # a cheap-thin-margin cluster from an expensive-thick-margin cluster so a
    # single contract does not have to compromise between them.
    s_sorted = sorted(servable, key=lambda t: t[0])
    ps = [t[0] for t in s_sorted]
    m = len(ps)
    if m >= 4:
        lo, hi = m // 10, m - m // 10
        gaps = [(ps[i + 1] - ps[i], i) for i in range(m - 1) if lo <= i <= hi]
        split = (max(gaps)[1] + 1) if gaps else m // 2
    else:
        split = m // 2
    group_a = s_sorted[:split]      # lower-risk tier (candidate for a cheap contract)
    group_b = s_sorted[split:]      # higher-risk tier (candidate for full coverage)

    contracts = []

    # tier B: exact sweep-optimal full-coverage price (monopolist threshold
    # search: profit is piecewise-linear in P with breakpoints at each -u_i,
    # so the optimum is always exactly at one of those breakpoints).
    p_b = None
    if group_b:
        cands = [(-u, p * L) for (p, a, L, u) in group_b]
        p_b, profit_b = sweep_optimal(cands)
        if profit_b > 0:
            contracts.append((max(0.0, p_b - 1e-6), 1.0))
        else:
            p_b = None

    # tier A: search coverage levels from most-separating (low c) to most-
    # extracting (high c); accounting for the risk-aversion variance penalty
    # of partial coverage, and keep only a level that is INCENTIVE-COMPATIBLE
    # -- i.e. the highest-margin member of tier B still strictly prefers its
    # own full-coverage contract over defecting to this cheap one. If nothing
    # clears both profitability and IC, tier A is deliberately left empty:
    # those agents are the ones the menu is designed to let walk away.
    if group_a:
        best = None
        for c in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            cands = []
            for (p, a, L, u) in group_a:
                resid = (1.0 - c) * L
                pm = -u - p * resid - a * p * (1.0 - p) * resid * resid
                cands.append((pm, p * c * L))
            p_a, profit_a = sweep_optimal(cands)
            if profit_a <= 0:
                continue
            if p_b is not None:
                ic_ok = True
                for (p, a, L, u) in group_b:
                    resid = (1.0 - c) * L
                    util_a = -(p_a + p * resid) - a * p * (1.0 - p) * resid * resid
                    if util_a > -p_b + 1e-6:
                        ic_ok = False
                        break
                if not ic_ok:
                    continue
            if best is None or profit_a > best[1]:
                best = (p_a, profit_a, c)
        if best is not None:
            contracts.append((max(0.0, best[0] - 1e-6), best[2]))

    print(len(contracts))
    for (P, c) in contracts:
        print("%.6f %.6f" % (P, c))


if __name__ == "__main__":
    main()
