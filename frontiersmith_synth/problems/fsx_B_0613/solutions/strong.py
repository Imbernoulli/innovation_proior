# TIER: strong
# INSIGHT: a sorting network only has to resolve the pairs the partial order leaves
# incomparable. Each planted run is an already-sorted block, so we NEVER compare
# within a run -- we only MERGE the runs. We recover the contiguous sorted runs from
# the relations, then merge them pairwise in a balanced tree using Batcher's general
# odd-even MERGE (standard min->low-index comparators). This deletes every comparator
# that would only re-verify a known relation.
import sys

def oemerge(x, y):
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return []
    if m == 1 and n == 1:
        u, v = x[0], y[0]
        return [(u, v)] if u < v else [(v, u)]
    ex, ox = x[0::2], x[1::2]
    ey, oy = y[0::2], y[1::2]
    comps = oemerge(ex, ey) + oemerge(ox, oy)
    W = sorted(x + y)
    for i in range(1, m + n - 1, 2):
        u, v = W[i], W[i + 1]
        comps.append((u, v) if u < v else (v, u))
    return comps

def main():
    toks = sys.stdin.read().split()
    n = int(toks[0]); E = int(toks[1])
    p = 3
    edgeset = set()
    for _ in range(E):
        a = int(toks[p]); b = int(toks[p + 1]); p += 2
        edgeset.add((a, b))
    # recover contiguous sorted runs: a boundary sits between k-1 and k
    # whenever (k-1,k) is NOT a known relation.
    runs = []
    start = 0
    for k in range(1, n):
        if (k - 1, k) not in edgeset:
            runs.append((start, k - start))
            start = k
    runs.append((start, n - start))

    comps = []
    groups = list(runs)  # each (start, length), contiguous & already sorted
    while len(groups) > 1:
        newg = []
        i = 0
        while i < len(groups):
            if i + 1 < len(groups):
                (s1, l1) = groups[i]
                (s2, l2) = groups[i + 1]
                x = list(range(s1, s1 + l1))
                y = list(range(s2, s2 + l2))
                comps += oemerge(x, y)
                newg.append((s1, l1 + l2))
                i += 2
            else:
                newg.append(groups[i]); i += 1
        groups = newg

    out = [str(len(comps))]
    for (a, b) in comps:
        out.append("%d %d" % (a, b))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
