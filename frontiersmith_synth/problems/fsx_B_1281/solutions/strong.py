# TIER: strong
import sys

def eff_tonnes(p):
    price, tonnes, reported, ref_base, irr, threshold, reversal, perm_years, buffer = p
    inflation_penalty = min(1.0, ref_base / reported)
    fin_add = (threshold - irr) / 1000.0
    fin_add = 0.0 if fin_add < 0.0 else (1.0 if fin_add > 1.0 else fin_add)
    additionality = min(inflation_penalty, fin_add)
    survival = (1.0 - reversal / 10000.0) ** perm_years
    permanence = 1.0 - (1.0 - buffer / 100.0) * (1.0 - survival)
    return tonnes * additionality * permanence

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    n = int(next(it)); budget = int(next(it))
    projects = []
    for _ in range(n):
        row = [int(next(it)) for _ in range(9)]
        projects.append(row)

    # Insight: price does not track true value. Re-derive each project's real,
    # additional, permanent tonnes from the counterfactual-evidence fields, then
    # solve the 0/1 knapsack on THAT value (not on price) under the dollar budget.
    costs = [projects[idx][0] * projects[idx][1] for idx in range(n)]
    values = [eff_tonnes(projects[idx]) for idx in range(n)]

    dp = [0.0] * (budget + 1)
    keep = [bytearray(budget + 1) for _ in range(n)]
    for j in range(n):
        c = costs[j]
        v = values[j]
        if c > budget:
            continue
        row = keep[j]
        dpc = dp  # local ref for speed
        for cap in range(budget, c - 1, -1):
            cand = dpc[cap - c] + v
            if cand > dpc[cap]:
                dpc[cap] = cand
                row[cap] = 1

    best_cap = max(range(budget + 1), key=lambda cap: dp[cap])
    chosen = []
    cap = best_cap
    for j in range(n - 1, -1, -1):
        if keep[j][cap]:
            chosen.append(j + 1)
            cap -= costs[j]

    print(len(chosen))
    print(" ".join(map(str, chosen)))

main()
