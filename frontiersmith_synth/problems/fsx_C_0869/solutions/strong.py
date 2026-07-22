# TIER: strong
"""The insight: this is a covering-style dual problem. Fix a candidate grade gh for a
hub feature h. That single choice simultaneously sets a private "budget"
  cap_c(gh) = min(spec_backup_c, spec_primary_c - tol(gh))
for EVERY chain c that shares h -- one unit of hub investment is reused by every chain
that touches it, for free. So instead of asking "what does this chain need from the
hub?" (the greedy question, answered chain-by-chain), ask "what hub grade minimizes
total spend across every chain that depends on this hub?" -- money should flow to a
feature in proportion to how many BINDING chains it serves, not to what any single
chain would ask for in isolation.

Because private features are chain-exclusive (never shared) and every chain has
exactly one hub, the total cost decomposes additively over hub groups: solving each
hub's mini cost-allocation problem independently and summing IS the global optimum for
this instance. For a fixed hub grade the private sub-problem (pick g1, g2 to respect a
tolerance-sum budget) is solved once via a small precomputed table."""
import sys

TOL = [64, 32, 16, 8, 4, 2, 1]
COST = [0, 1, 3, 7, 15, 31, 63]
GMAX = 6
MAXCAP = 3 * max(TOL) + 8  # generous upper bound on any spec_primary/backup we'll see


def build_priv_table(maxcap):
    """priv[cap] = (min cost(g1)+cost(g2), g1, g2) subject to tol(g1)+tol(g2) <= cap."""
    raw = {}
    for g1 in range(GMAX + 1):
        for g2 in range(GMAX + 1):
            t = TOL[g1] + TOL[g2]
            c = COST[g1] + COST[g2]
            if t not in raw or c < raw[t][0]:
                raw[t] = (c, g1, g2)
    table = [None] * (maxcap + 1)
    running = None
    for cap in range(maxcap + 1):
        if cap in raw and (running is None or raw[cap][0] < running[0]):
            running = raw[cap]
        table[cap] = running
    return table


PRIV = build_priv_table(MAXCAP)


def priv_best(cap):
    if cap < 0:
        return None
    if cap > MAXCAP:
        cap = MAXCAP
    return PRIV[cap]


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

    by_hub = {}
    for chain in chains:
        h = chain[2]
        by_hub.setdefault(h, []).append(chain)

    grade = [0] * m
    for h, lst in by_hub.items():
        best = None
        for gh in range(GMAX + 1):
            total = COST[gh]
            ok = True
            for (p1, p2, hh, sp, sb) in lst:
                cap = min(sb, sp - TOL[gh])
                pb = priv_best(cap)
                if pb is None:
                    ok = False
                    break
                total += pb[0]
            if ok and (best is None or total < best[0]):
                best = (total, gh)
        # best always exists at gh=GMAX given the generator's feasibility floor
        _, gh = best
        grade[h] = gh
        for (p1, p2, hh, sp, sb) in lst:
            cap = min(sb, sp - TOL[gh])
            _, g1, g2 = priv_best(cap)
            grade[p1] = g1
            grade[p2] = g2

    sys.stdout.write(" ".join(map(str, grade)) + "\n")


if __name__ == "__main__":
    main()
