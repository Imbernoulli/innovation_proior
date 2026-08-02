# TIER: strong
"""Insight: pure wirelength minimization is the right FIRST move, not the wrong
one -- the recipe fails only where it overflows a constraint, so don't discard
it. (1) Start from the exact same co-membership greedy chain the "greedy" tier
uses (so on any instance the recipe already satisfies, strong reproduces it
exactly, no wasted wirelength). (2) Only where that chain is infeasible, run a
bounded congestion-gradient descent: repeatedly find the WORST violated
resource (a timing-critical net over its slack, or the most-overloaded
channel), restrict candidate moves to cells that are not on any
timing-critical net, and relocate the candidate/target-slot pair that most
reduces total constraint violation (ties broken toward smaller wirelength
growth). This spreads exactly the cells that are safe to spread, and only as
far as needed, while keeping every timing-critical net short -- turning the
recipe's dense, congested core into a feasible layout instead of abandoning
it. A safety fallback to the identity placement guarantees the output is
never infeasible."""
import sys
from collections import defaultdict


def read_instance(text):
    it = iter(text.split())

    def nxt():
        return next(it)

    n_cells = int(nxt())
    n_nets = int(nxt())
    capacity = [int(nxt()) for _ in range(max(n_cells - 1, 0))]
    nets = []
    for _ in range(n_nets):
        k = int(nxt())
        crit = int(nxt()) == 1
        slack = int(nxt())
        terms = [int(nxt()) for _ in range(k)]
        nets.append({"terms": terms, "crit": crit, "slack": slack})
    return n_cells, n_nets, capacity, nets


def chain_order(n_cells, nets, crit_boost):
    w = defaultdict(int)
    for net in nets:
        terms = net["terms"]
        boost = crit_boost if net["crit"] else 1
        m = len(terms)
        for a in range(m):
            for b in range(a + 1, m):
                i, j = terms[a], terms[b]
                if i > j:
                    i, j = j, i
                w[(i, j)] += boost
    deg = [0] * n_cells
    for (i, j), val in w.items():
        deg[i] += val
        deg[j] += val
    placed = []
    placed_set = set()
    start = max(range(n_cells), key=lambda c: (deg[c], -c))
    placed.append(start)
    placed_set.add(start)
    conn = [0] * n_cells
    for j in range(n_cells):
        if j != start:
            key = (start, j) if start < j else (j, start)
            conn[j] += w.get(key, 0)
    while len(placed) < n_cells:
        best = None
        best_val = -1
        for c in range(n_cells):
            if c in placed_set:
                continue
            if conn[c] > best_val or (conn[c] == best_val and (best is None or c < best)):
                best_val = conn[c]
                best = c
        placed.append(best)
        placed_set.add(best)
        for j in range(n_cells):
            if j not in placed_set:
                key = (best, j) if best < j else (j, best)
                conn[j] += w.get(key, 0)
    return placed


def compute_state(nets, pos, n_cells):
    spans = [0] * len(nets)
    usage = [0] * max(n_cells - 1, 0)
    for idx, net in enumerate(nets):
        slots = [pos[c] for c in net["terms"]]
        lo, hi = min(slots), max(slots)
        spans[idx] = hi - lo
        for g in range(lo, hi):
            usage[g] += 1
    return spans, usage


def violation_score(nets, capacity, spans, usage):
    crit_v = sum(max(0, spans[i] - nets[i]["slack"]) for i in range(len(nets)) if nets[i]["crit"])
    cap_v = sum(max(0, usage[g] - capacity[g]) for g in range(len(capacity)))
    return crit_v, cap_v


def total_wl(spans):
    return sum(spans)


def main():
    text = sys.stdin.read()
    n_cells, n_nets, capacity, nets = read_instance(text)

    order = chain_order(n_cells, nets, crit_boost=1)
    pos = [0] * n_cells
    for slot, cell in enumerate(order):
        pos[cell] = slot

    critical_cells = set(c for net in nets if net["crit"] for c in net["terms"])

    best_feasible = None  # (wirelength, pos-copy) of the best feasible layout seen
    max_iters = max(40, 8 * n_cells)
    stall = 0
    prev_viol = None
    for it in range(max_iters):
        spans, usage = compute_state(nets, pos, n_cells)
        crit_v, cap_v = violation_score(nets, capacity, spans, usage)
        if crit_v == 0 and cap_v == 0:
            wl = total_wl(spans)
            if best_feasible is None or wl < best_feasible[0]:
                best_feasible = (wl, list(pos))
            break  # recipe (now repaired) is feasible: done

        if (crit_v, cap_v) == prev_viol:
            stall += 1
        else:
            stall = 0
        prev_viol = (crit_v, cap_v)

        if crit_v > 0:
            # the critical constraint is the hard one: fix it first, any cell eligible
            focus_cells = sorted(
                set(c for i, net in enumerate(nets) if net["crit"] and spans[i] > net["slack"]
                    for c in net["terms"])
            )
        else:
            worst_g = max(range(len(capacity)), key=lambda g: usage[g] - capacity[g])
            crossing = [
                net for net in nets
                if min(pos[c] for c in net["terms"]) <= worst_g < max(pos[c] for c in net["terms"])
            ]
            focus_cells = sorted(set(c for net in crossing for c in net["terms"]))

        # prefer relocating cells that carry no timing constraint; only touch a
        # critical-net cell if no purely-safe move can make progress
        non_crit_focus = [c for c in focus_cells if c not in critical_cells]
        candidate_tiers = [non_crit_focus, focus_cells] if non_crit_focus else [focus_cells]

        base_wl = total_wl(spans)
        best_move = None
        best_key = None
        for candidates in candidate_tiers:
            for c in candidates:
                c_slot = pos[c]
                for t_slot in range(n_cells):
                    if t_slot == c_slot:
                        continue
                    other = next(cc for cc in range(n_cells) if pos[cc] == t_slot)
                    pos[c], pos[other] = t_slot, c_slot  # tentative swap
                    n_spans, n_usage = compute_state(nets, pos, n_cells)
                    n_crit_v, n_cap_v = violation_score(nets, capacity, n_spans, n_usage)
                    n_wl = total_wl(n_spans)
                    key = (n_crit_v, n_cap_v, n_wl - base_wl)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_move = (c, other)
                    pos[c], pos[other] = c_slot, t_slot  # revert
            if best_key is not None and (best_key[0], best_key[1]) < (crit_v, cap_v):
                break  # this tier already found real progress; no need to widen further

        if best_move is None:
            break

        improves = best_key[:2] < (crit_v, cap_v)
        if not improves and stall >= 6:
            # stuck at a single-swap local optimum: deterministic perturbation to
            # escape (indices derived from the iteration counter -- no randomness)
            k1 = (it * 7 + 3) % n_cells
            k2 = (it * 13 + 5) % n_cells
            if k1 != k2:
                cell1 = next(cc for cc in range(n_cells) if pos[cc] == k1)
                cell2 = next(cc for cc in range(n_cells) if pos[cc] == k2)
                pos[cell1], pos[cell2] = pos[cell2], pos[cell1]
            stall = 0
            continue

        c, other = best_move
        pos[c], pos[other] = pos[other], pos[c]

    if best_feasible is not None:
        pos = best_feasible[1]
    else:
        pos = list(range(n_cells))  # safety fallback: identity is always feasible

    print(" ".join(map(str, pos)))


if __name__ == "__main__":
    main()
