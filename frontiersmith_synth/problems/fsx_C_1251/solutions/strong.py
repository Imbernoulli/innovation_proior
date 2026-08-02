# TIER: strong
"""The insight: raw frequency ignores that several candidate fusions can match the
EXACT same code region (mutually exclusive), so their true value is not additive --
selecting them all wastes encoding-space/area for zero extra benefit beyond the best
one. This is a set-cover-style problem: what matters is MARGINAL coverage gained per
unit area, not a candidate's standalone frequency. We run the textbook cost/benefit
greedy for weighted-coverage maximization under a knapsack (area) + cardinality
(encoding-space) budget: repeatedly add whichever still-affordable candidate yields the
largest marginal increase in cycles saved per unit area, simulating the SAME
deterministic single-pass fusing rule the checker uses, until nothing helps or the
budgets are exhausted."""
import sys


def main():
    data = sys.stdin.read().split("\n")
    idx = 0
    K, A = map(int, data[idx].split()); idx += 1
    C = int(data[idx]); idx += 1
    area = [0] * C
    size = [0] * C
    cost = [0] * C
    for c in range(C):
        a, s, cst = map(int, data[idx].split()); idx += 1
        area[c], size[c], cost[c] = a, s, cst
    M = int(data[idx]); idx += 1
    apps = []
    for m in range(M):
        L, O = map(int, data[idx].split()); idx += 1
        occs = []
        for _ in range(O):
            cid, start = map(int, data[idx].split()); idx += 1
            sv = size[cid] - cost[cid]
            occs.append((cid, start, sv, size[cid]))
        occs.sort(key=lambda o: (o[1], -o[2], -o[3], o[0]))
        apps.append((L, occs))

    def simulate(selected):
        total = 0
        for (L, occs) in apps:
            claimed = bytearray(L)
            for (cid, start, sv, sz) in occs:
                if cid not in selected:
                    continue
                free = True
                for p in range(start, start + sz):
                    if claimed[p]:
                        free = False
                        break
                if free:
                    for p in range(start, start + sz):
                        claimed[p] = 1
                    total += sv
        return total

    selected = set()
    used_area = 0
    base_val = 0
    remaining = set(range(C))
    while len(selected) < K and remaining:
        best_c, best_ratio, best_gain = None, -1.0, 0
        for c in list(remaining):
            if used_area + area[c] > A:
                remaining.discard(c)
                continue
            trial_val = simulate(selected | {c})
            gain = trial_val - base_val
            if gain <= 0:
                continue
            ratio = gain / max(1, area[c])
            if ratio > best_ratio + 1e-12 or (abs(ratio - best_ratio) <= 1e-12 and best_c is not None and area[c] < area[best_c]):
                best_c, best_ratio, best_gain = c, ratio, gain
        if best_c is None:
            break
        selected.add(best_c)
        used_area += area[best_c]
        base_val += best_gain
        remaining.discard(best_c)

    sel = sorted(selected)
    out = [str(len(sel)), " ".join(map(str, sel))]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
