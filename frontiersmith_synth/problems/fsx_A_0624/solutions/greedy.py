# TIER: greedy
# The obvious recipe: set capacities PROPORTIONAL TO FIRST-CHOICE DEMAND.
# Count how many apprentices list each workshop first, hand out T seats in that
# proportion (largest-remainder apportionment, capped by each workshop's room
# limit). Reasons purely about DEMAND COUNTS, never about where the deferred-
# acceptance rejection chains actually flow -- so a workshop nobody lists FIRST
# (even though everyone lists it 2nd/3rd) gets starved to near zero.
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); M = int(next(it)); T = int(next(it))
    cap_max = [int(next(it)) for _ in range(M)]
    demand = [0] * M
    for _ in range(N):
        Li = int(next(it))
        first = int(next(it))       # rank-1 workshop
        for _ in range(Li - 1):
            next(it)
        demand[first] += 1

    tot = sum(demand)
    caps = [0] * M
    if tot == 0:
        remaining = T
        j = 0
        while remaining > 0:
            if caps[j] < cap_max[j]:
                caps[j] += 1; remaining -= 1
            j = (j + 1) % M
    else:
        raw = [T * demand[w] / tot for w in range(M)]
        caps = [min(cap_max[w], int(raw[w])) for w in range(M)]
        remaining = T - sum(caps)
        # distribute the remainder by largest fractional share first, repeating
        # rounds (respecting each workshop's room limit) until T is reached.
        guard = 0
        while remaining > 0 and guard < 10 * M + 20:
            guard += 1
            cands = [w for w in range(M) if caps[w] < cap_max[w]]
            if not cands:
                break
            cands.sort(key=lambda w: (-(raw[w] - int(raw[w])), -demand[w], w))
            given = 0
            for w in cands:
                if remaining <= 0:
                    break
                caps[w] += 1; remaining -= 1; given += 1
            if given == 0:
                break

    sys.stdout.write("\n".join(map(str, caps)) + "\n")


if __name__ == "__main__":
    main()
