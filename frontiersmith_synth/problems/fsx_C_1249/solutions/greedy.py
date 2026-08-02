# TIER: greedy
# The "obvious" recipe: build a physically-local mesh. Rank every candidate
# link purely by its physical (ring-distance) cost, cheapest first, and keep
# adding links while the budget allows -- completely ignoring the traffic
# matrix. This is a textbook "spend the budget on the cheapest topology"
# heuristic; it fits under L_max by construction but has no idea where the
# actual hot traffic pairs are, so on planted non-uniform instances the hot
# pairs still take many hops or pile onto the same few backbone links.
import sys

def ring_cost(i, j, N):
    d = abs(i - j)
    return min(d, N - d)

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); L_max = int(next(it))
    next(it)  # CAP (unused by this strategy)
    next(it)  # STALL_COST (unused by this strategy)
    for _ in range(N * N):
        next(it)  # traffic matrix -- deliberately ignored

    candidates = []
    for i in range(N):
        for j in range(i + 1, N):
            candidates.append((ring_cost(i, j, N), i, j))
    candidates.sort()  # cheapest (most local) links first

    spent = 0
    edges = []
    for (c, i, j) in candidates:
        if spent + c <= L_max:
            edges.append((i, j))
            spent += c

    out = [str(len(edges))]
    for (u, v) in edges:
        out.append("%d %d" % (u, v))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
