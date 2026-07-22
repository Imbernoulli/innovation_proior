# TIER: greedy
# The natural first idea after "a constant level gives all-or-nothing": pick a
# single SWITCH TIME W and flip the dominance strength once (auxin high for
# the first W ticks, then released forever after). This is still a scalar
# parameter (a threshold TIME, not a real schedule), so we can afford to
# brute-force the best W directly. A single switch can only ever protect a
# PREFIX of buds -- it is structurally blind to any bud that must stay
# arrested after the switch, and it sacrifices every real target born before
# the switch. Search all budget-feasible W and keep the best.
import sys


def simulate(H, c, r):
    PS = [0] * (H + 1)
    for t in range(1, H + 1):
        PS[t] = PS[t - 1] + r[t - 1]
    act_time = {}
    for i in range(1, H + 1):
        if r[i - 1] == 0:
            act_time[i] = None
            continue
        need = PS[i - 1] + c[i - 1]
        if need <= PS[H]:
            lo, hi = i, H
            while lo < hi:
                mid = (lo + hi) // 2
                if PS[mid] >= need:
                    hi = mid
                else:
                    lo = mid + 1
            act_time[i] = lo
        else:
            act_time[i] = None
    activated = [(t, -i) for i, t in act_time.items() if t is not None]
    activated.sort()
    act_order = [-negi for (t, negi) in activated]
    return act_time, act_order


BETA2 = 0.15
ALPHA = 0.5


def score_topology(H, c, T, r):
    act_time, act_order = simulate(H, c, r)
    sim_active = set(i for i in act_time if act_time[i] is not None)
    Tset = set(T)
    correct = sim_active & Tset
    precision = (len(correct) / len(sim_active)) if sim_active else 0.0
    recall = (len(correct) / len(Tset)) if Tset else 1.0
    denom = BETA2 * precision + recall
    set_score = ((1 + BETA2) * precision * recall / denom) if denom > 0 else 0.0

    target_filtered = [x for x in T if x in correct]
    sim_rank = {node: idx for idx, node in enumerate(act_order)}
    m = len(target_filtered)
    if m <= 1:
        order_score = 1.0
    else:
        concordant = 0
        total = 0
        for a in range(m):
            for b in range(a + 1, m):
                total += 1
                if sim_rank[target_filtered[a]] < sim_rank[target_filtered[b]]:
                    concordant += 1
        order_score = concordant / total if total else 1.0
    return set_score * (ALPHA + (1 - ALPHA) * order_score)


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    H = int(next(it)); BUDGET = int(next(it)); K = int(next(it))
    c = [int(next(it)) for _ in range(H)]
    T = [int(next(it)) for _ in range(K)]

    best_F = -1.0
    best_r = [1] * H
    maxW = min(H, BUDGET)
    for W in range(0, maxW + 1):
        r = [0] * W + [1] * (H - W)
        F = score_topology(H, c, T, r)
        if F > best_F:
            best_F = F
            best_r = r
    sys.stdout.write(" ".join(str(x) for x in best_r) + "\n")


if __name__ == "__main__":
    main()
