# TIER: strong
#!/usr/bin/env python3
"""Insight: the metric that matters is survival-weighted capital, not raw
recovery-per-premium-dollar -- and survival is dominated by whether the layers
you buy form a CONTIGUOUS band (no hole for a loss to fall straight through),
not by which individual layer looks cheapest. Rather than ranking layers by
rate-on-line, re-simulate the *whole* scenario set (same fill/exhaust physics
as the checker: occurrence cap, aggregate reinstatement cap, ruin-on-breach)
for every contiguous WINDOW of the tower [lo..hi] that fits the budget, plus
a partial placement extending the window one layer further with whatever
premium is left over. Keep whichever contiguous program actually scores best
under simulation -- sometimes that is anchored at the retention (when the
scenario set threatens frequent low-band losses), sometimes it is anchored
higher up; the point is that the choice is verified by resimulating survival,
never assumed from a layer's price tag alone."""
import sys


def simulate(shares, catalog, scenarios, C0):
    M = len(catalog)
    RUIN_MULT = 3.0
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


def main():
    data = sys.stdin.read().split()
    it = iter(data)
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

    best = None
    best_shares = None
    for lo in range(M):
        for hi in range(lo, M):
            cost = sum(catalog[j][2] for j in range(lo, hi + 1))
            if cost > Pmax:
                continue
            shares = [0] * M
            for j in range(lo, hi + 1):
                shares[j] = 100
            leftover = Pmax - cost
            # try extending the contiguous window with the leftover premium,
            # either one band further up or one band further down
            for ext in (hi + 1, lo - 1):
                if 0 <= ext < M and shares[ext] == 0:
                    Prem = catalog[ext][2]
                    cand = shares[:]
                    cand[ext] = min(100, (100 * leftover) // Prem) if Prem > 0 else 0
                    val = simulate(cand, catalog, scenarios, C0)
                    if best is None or val > best:
                        best = val
                        best_shares = cand
            val = simulate(shares, catalog, scenarios, C0)
            if best is None or val > best:
                best = val
                best_shares = shares

    if best_shares is None:
        best_shares = [0] * M

    print(" ".join(map(str, best_shares)))


if __name__ == "__main__":
    main()
