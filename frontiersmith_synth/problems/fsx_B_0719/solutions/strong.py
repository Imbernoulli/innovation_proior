# TIER: strong
"""
The insight: the unit of optimization is the correlated PAIR-TRAIN, not
the lone item.

Item-frequency sorting is already the *provably optimal fixed* rack
order (rearrangement inequality) -- no relabeling trick beats it while
the rack never moves again, so a smart solver keeps that base order.
The real headroom is TEMPORAL, not positional: some pairs are not
individually frequent enough to earn a permanent front slot, yet their
real traffic is squeezed into one short, dense burst window. A single
paid round trip -- pull the pair to the front right before its burst,
then pay to put it straight back right after -- is charged only ONCE
per burst but discounts *every* access inside that burst for *both*
pair members, and leaves the rest of the day (and the truly hot items
that must not be evicted) completely undisturbed. Whether a given
train is worth that round trip is a genuine value/cost decision, run
as a knapsack over all pairs within the swap budget.
"""
import sys


def move_to_position(item, target, at_customer, order, pos, events):
    """Bubble `item` forward (only ever leftward) to slot `target`."""
    swaps = []
    p = pos[item]
    while p > target:
        j = p - 1
        u, v = order[j], order[j + 1]
        order[j], order[j + 1] = v, u
        pos[u], pos[v] = j + 1, j
        swaps.append(j)
        p -= 1
    for j in swaps:
        events.append((at_customer, j))
    return swaps


def pair_move_cost(a, b, pos):
    ca = max(0, pos[a] - 1)
    new_pos_b = pos[b] + 1 if pos[b] < pos[a] else pos[b]
    cb = max(0, new_pos_b - 2)
    return ca + cb


def main():
    data = sys.stdin.buffer.read().split()
    ptr = 0
    n = int(data[ptr]); ptr += 1
    m = int(data[ptr]); ptr += 1
    K = int(data[ptr]); ptr += 1
    P = int(data[ptr]); ptr += 1
    pairs = []
    for _ in range(P):
        a = int(data[ptr]); ptr += 1
        b = int(data[ptr]); ptr += 1
        pairs.append((a, b))
    seq = [int(x) for x in data[ptr:ptr + m]]

    freq = [0] * (n + 1)
    for x in seq:
        freq[x] += 1

    # 1) base order: the provably optimal STATIC order (descending raw
    #    frequency) -- identical starting point to the textbook approach.
    init_order = sorted(range(1, n + 1), key=lambda it: (-freq[it], it))
    order = [0] + init_order[:]
    pos = [0] * (n + 1)
    for slot in range(1, n + 1):
        pos[order[slot]] = slot

    idx_of_pair = {}
    for a, b in pairs:
        idx_of_pair[a] = (a, b)
        idx_of_pair[b] = (a, b)
    occ = {p: [] for p in pairs}
    for i, x in enumerate(seq, 1):
        if x in idx_of_pair:
            occ[idx_of_pair[x]].append(i)

    # 2) value/cost estimate for a full round-trip rescue of each pair's
    #    burst window, using the STATIC positions to rank candidates.
    candidates = []
    for a, b in pairs:
        idxs = occ[(a, b)]
        if not idxs:
            continue
        run_start, run_end = idxs[0], idxs[-1]
        benefit = 0
        for i in idxs:
            item = seq[i - 1]
            tgt = 1 if item == a else 2
            benefit += pos[item] - tgt
        one_way = pair_move_cost(a, b, pos)
        round_trip_cost = 2 * one_way if run_end < m else one_way
        ratio = benefit / max(1, round_trip_cost)
        candidates.append((ratio, round_trip_cost, run_start, run_end, a, b))

    # 3) knapsack selection within the swap budget, by value/cost.
    candidates.sort(key=lambda c: -c[0])
    budget_left_est = K
    selected = []
    for ratio, cost_est, run_start, run_end, a, b in candidates:
        if ratio <= 0:
            continue
        if cost_est <= budget_left_est:
            selected.append((run_start, run_end, a, b))
            budget_left_est -= cost_est

    # 4) execute in time order: forward move before the burst, backward
    #    move (exact undo) right after it -- epochs are disjoint in time
    #    so each pair fully returns the rack to baseline before the next.
    selected.sort(key=lambda s: s[0])
    events = []
    budget_left = K
    for run_start, run_end, a, b in selected:
        need_fwd = pair_move_cost(a, b, pos)
        need_bwd = need_fwd if run_end < m else 0
        if need_fwd + need_bwd > budget_left:
            continue
        fwd_a = move_to_position(a, 1, run_start, order, pos, events)
        fwd_b = move_to_position(b, 2, run_start, order, pos, events)
        used = len(fwd_a) + len(fwd_b)
        if run_end < m:
            back_at = run_end + 1
            # undo: replay the exact same swap slots in reverse order
            undo_slots = list(reversed(fwd_b)) + list(reversed(fwd_a))
            for j in undo_slots:
                u, v = order[j], order[j + 1]
                order[j], order[j + 1] = v, u
                pos[u], pos[v] = j + 1, j
                events.append((back_at, j))
                used += 1
        budget_left -= used

    events.sort(key=lambda e: e[0])

    out = []
    out.append(" ".join(str(x) for x in init_order))
    out.append(str(len(events)))
    for ci, cj in events:
        out.append("%d %d" % (ci, cj))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
