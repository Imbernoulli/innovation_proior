# TIER: greedy
"""The standard tolerance-allocation recipe: solve each requirement chain COMPLETELY
INDEPENDENTLY of every other chain (as if it were the only chain in the system), find
that chain's own cost-optimal grade split among its 3 members (this already beats the
'trivial' per-check-fair-share rule, since within one chain it correctly trades private
vs. hub investment) -- then merge the C per-chain answers by taking, for every feature,
the MAX grade any chain asked of it.

What this recipe never does: notice that a hub feature is reused by dozens of OTHER
chains too. Each chain, evaluated alone, only "sees" its own payoff from upgrading the
hub, so it stops investing there as soon as it stops paying off for itself -- even
though the same upgrade would keep paying off across every other chain sharing that
hub. That is exactly the trap: independently-optimal decisions do not aggregate to a
globally optimal one when the decisions share a resource."""
import sys

TOL = [64, 32, 16, 8, 4, 2, 1]
COST = [0, 1, 3, 7, 15, 31, 63]
GMAX = 6


def best_for_chain(sp, sb):
    """Brute-force the cost-minimal (g1, g2, gh) for THIS chain alone."""
    best = None
    for g1 in range(GMAX + 1):
        t1 = TOL[g1]
        for g2 in range(GMAX + 1):
            t2 = TOL[g2]
            if t1 + t2 > sb:
                continue
            base = t1 + t2
            c12 = COST[g1] + COST[g2]
            for gh in range(GMAX + 1):
                if base + TOL[gh] > sp:
                    continue
                c = c12 + COST[gh]
                if best is None or c < best[0]:
                    best = (c, g1, g2, gh)
    return best


def main():
    data = sys.stdin.read().split()
    idx = 0
    m = int(data[idx]); idx += 1
    C = int(data[idx]); idx += 1
    chains = []
    for _ in range(C):
        p1 = int(data[idx]); idx += 1
        p2 = int(data[idx]); idx += 1
        h = int(data[idx]); idx += 1
        sp = int(data[idx]); idx += 1
        sb = int(data[idx]); idx += 1
        chains.append((p1, p2, h, sp, sb))

    grade = [0] * m
    for (p1, p2, h, sp, sb) in chains:
        _, g1, g2, gh = best_for_chain(sp, sb)
        grade[p1] = g1
        grade[p2] = g2
        grade[h] = max(grade[h], gh)  # "take per-feature max grades" merge rule

    sys.stdout.write(" ".join(map(str, grade)) + "\n")


if __name__ == "__main__":
    main()
