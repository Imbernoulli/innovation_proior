# TIER: strong
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); M = int(next(it)); R = int(next(it)); K = int(next(it))
    values = [int(next(it)) for _ in range(N)]
    cap = [0] + [int(next(it)) for _ in range(M)]

    # INSIGHT: a shard with replicas on exactly two distinct machines {p1,p2} dies for
    # ONE machine-pair disaster only (namely (p1,p2)) -- it survives every other pair,
    # because a disaster removes just two machines and one of the shard's two replicas
    # always lands outside it. Percentile (not minimax) scoring means we don't have to
    # defend (p1,p2) at all: with K>1 we may write off the K-1 worst outcomes, so we
    # deliberately concentrate that one unavoidable sacrifice onto a SINGLE designated
    # pair, chosen as the two highest-capacity machines (cheapest to pack full replicas
    # onto). That buys near-total immunity for 2 replicas/shard instead of the 3
    # replicas/shard true immunity would cost -- freeing budget to cover more shards.
    machs = sorted(range(1, M + 1), key=lambda m: (-cap[m], m))
    p1, p2 = machs[0], machs[1]

    order = sorted(range(1, N + 1), key=lambda i: -values[i - 1])

    remaining = cap[:]
    budget = R
    pairs = []
    covered = [False] * (N + 1)

    # Phase A: cheap "immune except (p1,p2)" protection for as many top-value shards
    # as the sacrifice-pair's own capacity and the budget allow.
    for t in order:
        if budget < 2 or remaining[p1] <= 0 or remaining[p2] <= 0:
            break
        pairs.append((t, p1))
        pairs.append((t, p2))
        remaining[p1] -= 1
        remaining[p2] -= 1
        budget -= 2
        covered[t] = True

    # Phase B: leftover budget buys single-replica coverage on OTHER machines (kept
    # off p1/p2 so we don't smear extra vulnerability onto the (p1,x)/(p2,x) pairs
    # that are otherwise pristine), least-loaded machine first, still value order.
    def least_loaded_other():
        best = None
        best_frac = None
        for m in range(1, M + 1):
            if m in (p1, p2) or remaining[m] <= 0:
                continue
            frac = (cap[m] - remaining[m]) / cap[m]
            if best is None or frac < best_frac or (frac == best_frac and m < best):
                best = m
                best_frac = frac
        return best

    for t in order:
        if covered[t]:
            continue
        if budget <= 0:
            break
        m = least_loaded_other()
        if m is None:
            if remaining[p1] > 0:
                m = p1
            elif remaining[p2] > 0:
                m = p2
            else:
                continue
        pairs.append((t, m))
        remaining[m] -= 1
        budget -= 1
        covered[t] = True

    out = [str(len(pairs))]
    for (t, m) in pairs:
        out.append("%d %d" % (t, m))
    sys.stdout.write("\n".join(out) + "\n")


main()
