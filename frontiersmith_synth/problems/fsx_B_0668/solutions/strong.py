# TIER: strong
# Insight: only the ORDER in which lots draw on a bidder's budget matters, and only
# CONTESTED lots (>=2 strong bidders) meaningfully drain a winner's budget (an
# uncontested win just costs the low floor price). So (1) build a bidder-overlap
# graph from each lot's top-2 bidders, (2) group lots by their dominant bidder and
# interleave (round-robin) across groups so no single rich bidder's wins are bunched
# consecutively -- spreading its budget-depletion trajectory instead of burning it in
# one region and starving contested lots elsewhere, (3) seed reserves near each lot's
# second-highest bid (a mild monopoly bump on uncontested lots), then (4) hill-climb
# swaps/moves/reserve-deltas directly against the true replay objective under a
# wall-clock budget.
import sys
import time
import random


def simulate(n, m, values, budgets, pairs):
    remaining = budgets[:]
    revenue = 0
    for lot0, r in pairs:
        row = values[lot0]
        best_bid = -1
        best_j = -1
        second_bid = -1
        for j in range(m):
            v = row[j]
            if v <= 0:
                continue
            rem = remaining[j]
            b = v if v < rem else rem
            if b < r:
                continue
            if b > best_bid:
                second_bid = best_bid
                best_bid = b
                best_j = j
            elif b > second_bid:
                second_bid = b
        if best_j < 0:
            continue
        price = r if second_bid < 0 else (second_bid if second_bid > r else r)
        revenue += price
        remaining[best_j] -= price
    return revenue


def main():
    t_start = time.monotonic()
    DEADLINE = 3.6

    toks = sys.stdin.read().split()
    it = iter(toks)
    n = int(next(it)); m = int(next(it))
    values = [[int(next(it)) for _ in range(m)] for _ in range(n)]
    budgets = [int(next(it)) for _ in range(m)]

    # top-2 bidders per lot (single pass, no full sort).
    best1v = [0] * n; best1j = [-1] * n
    best2v = [0] * n
    for i in range(n):
        b1v = -1; b1j = -1; b2v = -1
        row = values[i]
        for j in range(m):
            v = row[j]
            if v <= 0:
                continue
            if v > b1v:
                b2v = b1v; b1v = v; b1j = j
            elif v > b2v:
                b2v = v
        best1v[i] = max(b1v, 0); best1j[i] = b1j
        best2v[i] = max(b2v, 0)

    # Group lots by their dominant (top) bidder -- each bidder's own "spend" block.
    # Order groups by ASCENDING average value: a bidder that appears as a valuable
    # *underbidder* on a cheaper block elsewhere should get to play that protective
    # role BEFORE it spends its own budget winning its (pricier) dominant block --
    # selling cheap-block collisions first keeps that bidder solvent for them.
    # Round-robin across groups interleaves different bidders' spending so no single
    # bidder's budget is burned in one consecutive run at another's expense.
    groups = {}
    for i in range(n):
        groups.setdefault(best1j[i], []).append(i)
    for j in groups:
        groups[j].sort(key=lambda i: best1v[i])
    group_keys = sorted(groups.keys(), key=lambda j: sum(best1v[i] for i in groups[j]) / len(groups[j]))

    order = []
    ptrs = {j: 0 for j in group_keys}
    remaining_lots = n
    while remaining_lots > 0:
        for j in group_keys:
            p = ptrs[j]
            lst = groups[j]
            if p < len(lst):
                order.append(lst[p])
                ptrs[j] = p + 1
                remaining_lots -= 1

    # initial reserves: for genuinely contested lots, the natural second-price value
    # is already the right reserve. For uncontested lots (second bidder is a cheap
    # filler, far below the top bid) the natural clearing price would collapse to
    # that filler's floor -- push the reserve up near the dominant bidder's own true
    # value instead, monopoly-style, to capture most of what they're actually
    # willing to pay (order/local search below then makes sure that bidder's budget
    # is still intact when this lot comes up).
    reserve = [0] * n
    for i in range(n):
        b1, b2 = best1v[i], best2v[i]
        if b2 < 0.3 * b1:
            reserve[i] = int(0.85 * b1)
        else:
            reserve[i] = b2

    def score(ordr, res):
        pairs = [(ordr[k], res[ordr[k]]) for k in range(n)]
        return simulate(n, m, values, budgets, pairs)

    best_order = order[:]
    best_reserve = reserve[:]
    best_score = score(best_order, best_reserve)

    cur_order = best_order[:]
    cur_reserve = best_reserve[:]
    cur_score = best_score

    rng = random.Random(1234567)

    while time.monotonic() - t_start < DEADLINE:
        move = rng.random()
        cand_order = cur_order
        cand_reserve = cur_reserve
        touched = False
        if move < 0.4:
            # swap two positions
            a, b = rng.randrange(n), rng.randrange(n)
            if a != b:
                cand_order = cur_order[:]
                cand_order[a], cand_order[b] = cand_order[b], cand_order[a]
                touched = True
        elif move < 0.7:
            # move one lot to another position
            a, b = rng.randrange(n), rng.randrange(n)
            if a != b:
                cand_order = cur_order[:]
                lot = cand_order.pop(a)
                cand_order.insert(b, lot)
                touched = True
        else:
            # perturb a lot's reserve
            lot = rng.randrange(n)
            b1 = best1v[lot]
            if b1 > 0:
                delta = max(1, b1 // 10)
                cand_reserve = cur_reserve[:]
                nv = cand_reserve[lot] + rng.choice([-delta, delta])
                nv = max(0, min(5_000_000, nv))
                cand_reserve[lot] = nv
                touched = True

        if not touched:
            continue

        cand_score = score(cand_order, cand_reserve)
        if cand_score >= cur_score:
            cur_order, cur_reserve, cur_score = cand_order, cand_reserve, cand_score
            if cur_score > best_score:
                best_order, best_reserve, best_score = cur_order[:], cur_reserve[:], cur_score
        # occasional restart from the best-known if we've drifted (sideways moves can
        # accumulate); cheap safety valve.
        elif rng.random() < 0.02:
            cur_order, cur_reserve, cur_score = best_order[:], best_reserve[:], best_score

    out = []
    for k in range(n):
        lot0 = best_order[k]
        out.append(f"{lot0 + 1} {best_reserve[lot0]}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
