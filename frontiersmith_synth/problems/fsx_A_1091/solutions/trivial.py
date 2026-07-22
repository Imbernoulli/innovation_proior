# TIER: trivial
# Baseline: reproduce the checker's documented canonical seeded-random
# blurring attack exactly -> scores Ratio = 0.1 per case by construction.
import sys
import random


def canonical_attack(n, edges, budget):
    rng = random.Random(1000003 + n * 131 + budget)
    edge_set = set(edges)
    elist = list(edges)
    swaps = []
    attempts = 0
    limit = 60 * budget + 200
    while len(swaps) < budget and attempts < limit:
        attempts += 1
        i = rng.randrange(len(elist))
        j = rng.randrange(len(elist))
        if i == j:
            continue
        a, b = elist[i]
        c, d = elist[j]
        if a == c or a == d or b == c or b == d:
            continue
        e1 = (a, d) if a < d else (d, a)
        e2 = (b, c) if b < c else (c, b)
        if e1 == e2 or e1 in edge_set or e2 in edge_set:
            continue
        edge_set.discard((a, b))
        edge_set.discard((c, d))
        edge_set.add(e1)
        edge_set.add(e2)
        elist[i] = e1
        elist[j] = e2
        swaps.append((a, b, c, d))
    return swaps


def main():
    tok = sys.stdin.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        t = int(tok[pos]); pos += 1
        return t

    n = nxt(); m = nxt(); k = nxt(); budget = nxt()
    labels = [nxt() for _ in range(n)]
    edges = [(nxt(), nxt()) for _ in range(m)]
    edges = sorted({(min(a, b), max(a, b)) for (a, b) in edges})
    swaps = canonical_attack(n, edges, budget)
    out = [str(len(swaps))]
    out += ["%d %d %d %d" % s for s in swaps]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
