# TIER: greedy
# The obvious recipe: "blur the wards". Spend the whole budget converting
# within-ward edges into cross-ward edges, preferring the highest-degree
# endpoints (move the most edge mass across the boundary). Structure-blind:
# it never looks at the eigenvectors, so on well-separated towns the noise
# is spectrally incoherent and the informative eigenvectors barely rotate.
import sys


def main():
    tok = sys.stdin.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        t = int(tok[pos]); pos += 1
        return t

    n = nxt(); m = nxt(); k = nxt(); budget = nxt()
    labels = [nxt() for _ in range(n)]
    raw = [(nxt(), nxt()) for _ in range(m)]
    edges = sorted({(min(a, b), max(a, b)) for (a, b) in raw if a != b})

    deg = [0] * n
    for (a, b) in edges:
        deg[a] += 1
        deg[b] += 1

    # within-community edges, sorted by descending degree product
    within = [( -deg[a] * deg[b], a, b) for (a, b) in edges if labels[a] == labels[b]]
    within.sort()
    # per-community queues of candidate edges (as (a,b))
    from collections import defaultdict
    queues = defaultdict(list)
    for (_, a, b) in within:
        queues[labels[a]].append((a, b))
    # community order: deterministic cyclic pairing c -> (c+1) % k
    ptr = defaultdict(int)
    es = set(edges)
    swaps = []
    used = set()

    def take(c):
        while ptr[c] < len(queues[c]):
            e = queues[c][ptr[c]]
            ptr[c] += 1
            if e in es and e not in used:
                return e
        return None

    order = sorted(queues.keys())
    stalled = 0
    ci = 0
    while len(swaps) < budget and stalled < 4 * len(order) + 4 and order:
        c = order[ci % len(order)]
        q = order[(ci + 1) % len(order)]
        ci += 1
        if c == q:
            continue
        e1 = take(c)
        if e1 is None:
            stalled += 1
            continue
        e2 = take(q)
        if e2 is None:
            # put e1 back is impossible (already consumed); just count stall
            stalled += 1
            continue
        a, b = e1
        c2, d = e2
        if len({a, b, c2, d}) != 4:
            continue
        n1 = (min(a, d), max(a, d))
        n2 = (min(b, c2), max(b, c2))
        if n1 == n2 or n1 in es or n2 in es:
            continue
        es.discard((a, b))
        es.discard((c2, d))
        es.add(n1)
        es.add(n2)
        used.add((a, b))
        used.add((c2, d))
        swaps.append((a, b, c2, d))
        stalled = 0

    out = [str(len(swaps))]
    out += ["%d %d %d %d" % s for s in swaps]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
