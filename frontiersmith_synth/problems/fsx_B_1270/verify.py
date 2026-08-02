#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for the reinsurance-layer-buy
problem. Prints 'Ratio: <float in [0,1]>' on its own final line.

Simulates the cedant's capital through each policy-year scenario in chronological
event order. For every event of loss X, each purchased layer j (participation
share p_j/100) recovers:
    occurrence-capped:  occ = min(max(X - A_j, 0), W_j)          (layer-attachment
                                                                    -exhaustion)
    aggregate-capped:   actual = min(occ * share_j, remaining_capacity_j)
        where remaining_capacity_j starts the year at
        share_j * (K_j + 1) * W_j                                (aggregate-vs
        and is drawn down by every recovery, resetting only next year -occurrence,
                                                                    reinstatement
                                                                    -limits)
A reinstatement cost (RP_j% of the amount just recovered) is charged from capital
immediately. If capital ever goes negative, the cedant is ruined for that
scenario: the year stops there and the outcome is 3x the (negative) capital at
the moment of breach -- a harsher, continuous penalty for a deeper breach,
reflecting that insolvency costs scale with the size of the shortfall.
"""
import sys
import math

RUIN_MULT = 3.0


def simulate(shares, catalog, scenarios, C0):
    M = len(catalog)
    prem_spent = sum(shares[j] / 100.0 * catalog[j][2] for j in range(M))
    outcomes = []
    for scen in scenarios:
        capital = C0 - prem_spent
        remcap = [shares[j] / 100.0 * (catalog[j][3] + 1) * catalog[j][1] for j in range(M)]
        breach = None
        for X in scen:
            total_rec = 0.0
            total_reinst = 0.0
            for j in range(M):
                if shares[j] == 0:
                    continue
                A, W, Prem, K, RP = catalog[j]
                occ = min(max(X - A, 0), W)
                occ_share = occ * (shares[j] / 100.0)
                actual = min(occ_share, remcap[j])
                remcap[j] -= actual
                total_rec += actual
                total_reinst += actual * (RP / 100.0)
            retained = X - total_rec
            capital -= retained
            capital -= total_reinst
            if capital < 0:
                breach = capital
                break
        outcomes.append(breach * RUIN_MULT if breach is not None else capital)
    return sum(outcomes) / len(outcomes)


def fail(msg):
    print("Ratio: 0.0  (%s)" % msg)
    return 0


def main():
    if len(sys.argv) < 3:
        return fail("bad args")
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path) as f:
        itoks = f.read().split()
    it = iter(itoks)
    try:
        M = int(next(it))
        catalog = []
        for _ in range(M):
            A = int(next(it)); W = int(next(it)); Prem = int(next(it))
            K = int(next(it)); RP = int(next(it))
            catalog.append((A, W, Prem, K, RP))
        C0 = int(next(it))
        Pmax = int(next(it))
        S = int(next(it))
        scenarios = []
        for _ in range(S):
            n = int(next(it))
            evs = [int(next(it)) for _ in range(n)]
            scenarios.append(evs)
    except StopIteration:
        return fail("truncated input")

    try:
        with open(out_path) as f:
            otoks = f.read().split()
    except FileNotFoundError:
        return fail("no output file")

    if len(otoks) != M:
        return fail("expected %d tokens, got %d" % (M, len(otoks)))

    shares = []
    for tok in otoks:
        try:
            val = int(tok)
        except ValueError:
            return fail("non-integer token %r" % tok)
        if not math.isfinite(val):
            return fail("non-finite token %r" % tok)
        if val < 0 or val > 100:
            return fail("participation share out of [0,100]: %d" % val)
        shares.append(val)

    cost100 = sum(shares[j] * catalog[j][2] for j in range(M))
    if cost100 > 100 * Pmax:
        return fail("premium budget exceeded: %.2f > %d" % (cost100 / 100.0, Pmax))

    F = simulate(shares, catalog, scenarios, C0)

    # internal baseline: spread the SAME premium budget evenly (equal % share)
    # over every layer in the catalog
    total_prem = sum(c[2] for c in catalog)
    p = min(100, (100 * Pmax) // total_prem) if total_prem > 0 else 0
    base_shares = [int(p)] * M
    B = simulate(base_shares, catalog, scenarios, C0)

    if B <= 0:
        return fail("degenerate instance: baseline achieves non-positive value")

    sc = max(0.0, min(1000.0, 100.0 * F / B))
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
