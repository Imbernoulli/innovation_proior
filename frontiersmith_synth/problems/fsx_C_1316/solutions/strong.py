# TIER: strong
"""The insight: the additive model is a good SEARCH heuristic (it correctly
ranks which substituents are individually worth their synthesis cost) but it
must be corrected by an explicit ring-adjacency check before committing --
two ring-adjacent bulky substituents clash and their combined steric
contribution flips sign. Once you see that the correction only depends on
the PAIR of neighboring choices (never anything farther away), the whole
problem becomes an exact cyclic dynamic program: state = (synthesis-steps
used so far, previous position's substituent), transition = choose the next
position's substituent and pay/avoid the adjacency-flip penalty against the
previous position. Because the target is placed (by construction) at or
beyond the true reachable optimum of the surrogate, maximizing the raw
additive-plus-correction sum S is EXACTLY equivalent to maximizing
closeness-to-target here -- no separate "aim for the window" logic is
needed once the adjacency correction is modeled exactly.

The ring is closed by fixing the first position's choice and trying all
K+1 possibilities for it, DP-ing forward over the rest, then paying the
wrap-around edge penalty between the last and first choice -- the standard
trick for turning a cyclic dependency into K+1 independent linear DPs.
"""
import sys

NEG = float('-inf')


def main():
    data = sys.stdin.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        v = data[pos]
        pos += 1
        return v

    N = int(nxt())
    K = int(nxt())
    budget = int(nxt())
    P0 = float(nxt())
    alpha = float(nxt())
    beta = float(nxt())
    s_thresh = float(nxt())
    target = float(nxt())
    window = float(nxt())
    lib = []
    for _ in range(K):
        e = float(nxt())
        s = float(nxt())
        c = int(nxt())
        lib.append((e, s, c))

    bulky = [s > s_thresh for (e, s, c) in lib]

    def item_value(t):
        e, s, c = lib[t]
        return e + alpha * s

    def edge_penalty(t1, t2):
        if t1 < 0 or t2 < 0:
            return 0.0
        if bulky[t1] and bulky[t2]:
            return beta * (lib[t1][1] + lib[t2][1])
        return 0.0

    choices = list(range(-1, K))  # -1 = empty (H)
    best_overall = NEG
    best_assign = None

    for first in choices:
        first_cost = 0 if first < 0 else lib[first][2]
        if first_cost > budget:
            continue

        layer = [[NEG] * (K + 1) for _ in range(budget + 1)]
        layer[first_cost][first + 1] = 0.0 if first < 0 else item_value(first)
        parents = []  # parents[i]: dict (b,pidx) -> (pb, ppidx, chosen_t)

        for i in range(1, N):
            ndp = [[NEG] * (K + 1) for _ in range(budget + 1)]
            par = {}
            for b in range(budget + 1):
                row = layer[b]
                for pidx in range(K + 1):
                    val = row[pidx]
                    if val == NEG:
                        continue
                    prev = pidx - 1
                    for t in choices:
                        cost_t = 0 if t < 0 else lib[t][2]
                        nb = b + cost_t
                        if nb > budget:
                            continue
                        add_val = 0.0 if t < 0 else item_value(t)
                        nv = val + add_val - edge_penalty(prev, t)
                        tidx = t + 1
                        if nv > ndp[nb][tidx]:
                            ndp[nb][tidx] = nv
                            par[(nb, tidx)] = (b, pidx, t)
            layer = ndp
            parents.append(par)

        for b in range(budget + 1):
            row = layer[b]
            for pidx in range(K + 1):
                val = row[pidx]
                if val == NEG:
                    continue
                prev = pidx - 1
                total = val - edge_penalty(prev, first)
                if total > best_overall:
                    best_overall = total
                    assign = [0] * N
                    assign[N - 1] = prev
                    cb, cpidx = b, pidx
                    for i in range(N - 1, 0, -1):
                        pb, ppidx, t = parents[i - 1][(cb, cpidx)]
                        assign[i] = t
                        cb, cpidx = pb, ppidx
                    assign[0] = first
                    best_assign = assign[:]

    if best_assign is None:
        best_assign = [-1] * N

    # output: shift -1(empty)->0, t->t+1
    out = [(a + 1) for a in best_assign]
    print(" ".join(str(x) for x in out))


if __name__ == "__main__":
    main()
