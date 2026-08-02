# TIER: greedy
"""The obvious first approach: minimize wirelength alone. Build a co-membership
weight between every pair of cells that share a net, then greedily grow a single
chain by always appending the unplaced cell most strongly connected to what has
already been placed (a standard minimum-linear-arrangement heuristic). This
completely ignores channel capacity and timing slack -- it only cares about
making nets short. On sparse instances that is harmless and wins big; on the
dense "hub net" instances it crams everything into one small block and blows
straight through the channel capacities there (Ratio: 0.0)."""
import sys
from collections import defaultdict


def main():
    data = sys.stdin.read().split()
    it = iter(data)

    def nxt():
        return next(it)

    n_cells = int(nxt())
    n_nets = int(nxt())
    for _ in range(max(n_cells - 1, 0)):
        nxt()  # capacities: ignored by design (pure wirelength greed)
    nets = []
    for _ in range(n_nets):
        k = int(nxt())
        nxt()  # crit: ignored
        nxt()  # slack: ignored
        terms = [int(nxt()) for _ in range(k)]
        nets.append(terms)

    w = defaultdict(int)
    for terms in nets:
        m = len(terms)
        for a in range(m):
            for b in range(a + 1, m):
                i, j = terms[a], terms[b]
                if i > j:
                    i, j = j, i
                w[(i, j)] += 1

    deg = [0] * n_cells
    for (i, j), val in w.items():
        deg[i] += val
        deg[j] += val

    placed = []
    placed_set = set()
    start = max(range(n_cells), key=lambda c: (deg[c], -c))
    placed.append(start)
    placed_set.add(start)
    conn = [0] * n_cells
    for j in range(n_cells):
        if j != start:
            key = (start, j) if start < j else (j, start)
            conn[j] += w.get(key, 0)

    while len(placed) < n_cells:
        best = None
        best_val = -1
        for c in range(n_cells):
            if c in placed_set:
                continue
            if conn[c] > best_val or (conn[c] == best_val and (best is None or c < best)):
                best_val = conn[c]
                best = c
        placed.append(best)
        placed_set.add(best)
        for j in range(n_cells):
            if j not in placed_set:
                key = (best, j) if best < j else (j, best)
                conn[j] += w.get(key, 0)

    pos = [0] * n_cells
    for slot, cell in enumerate(placed):
        pos[cell] = slot
    print(" ".join(map(str, pos)))


if __name__ == "__main__":
    main()
