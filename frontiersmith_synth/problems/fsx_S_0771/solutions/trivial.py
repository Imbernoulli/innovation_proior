# TIER: trivial
"""Reproduces the checker's own weak baseline: for each deck node, build
ONLY the straight vertical chain directly beneath it down to the anchor
(no diagonals, no bracing). These chains never share a member, so they are
always self-weight feasible on their own; output them bottom-up (anchor
first) so the build order is trivially safe too."""
import sys


def main():
    toks = sys.stdin.read().split()
    p = 0
    W, H, M, D = (int(toks[p + i]) for i in range(4)); p += 4
    edges = []
    for _ in range(M):
        u, v, c = int(toks[p]), int(toks[p + 1]), int(toks[p + 2]); p += 3
        edges.append((u, v, c))
    deck = [int(toks[p + i]) for i in range(D)]; p += D

    lookup = {}
    for idx, (u, v, c) in enumerate(edges):
        lookup[(u, v)] = idx
        lookup[(v, u)] = idx

    def node_id(x, y):
        return y * (W + 1) + x

    chosen = []
    seen = set()
    for d in deck:
        x = d % (W + 1)
        y = d // (W + 1)
        # bottom-up: anchor riser first, then progressively higher risers
        levels = list(range(y, 0, -1))
        for lvl in reversed(levels):  # lvl from 1 up to y -> build low-to-high
            u, v = node_id(x, lvl - 1), node_id(x, lvl)
            key = (u, v) if (u, v) in lookup else (v, u)
            idx = lookup[key]
            if idx not in seen:
                seen.add(idx)
                chosen.append(idx)

    out = [str(len(chosen))]
    out.extend(str(i) for i in chosen)
    print("\n".join(out))


if __name__ == "__main__":
    main()
