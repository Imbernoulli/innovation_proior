#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE truss-build instance to stdout.

Grid: x in [0,W], y in [0,H]. Node id = y*(W+1)+x. Nodes with id in [0,W] (y=0)
are anchors (foundation). Candidate members are "risers" (y -> y+1, dx in
{-1,0,1}) and "braces" (horizontal, same y>=1). Each candidate member has an
integer capacity. Deck nodes (y=H) are listed explicitly.

Difficulty ladder (testId 1..10, small -> large/adversarial):
  1-2: H=1, single level (always ordering-safe: every member touches an anchor).
  3-5: H=2, neutral (uniform) capacities -- may or may not trap a naive builder.
  6-10: H=2/3, TRAP capacities: the direct anchor risers (y=0->1) are
        deliberately WEAK while everything built on top of them (higher risers
        and horizontal braces) is deliberately STRONG. A builder who adds
        members in order of decreasing capacity (an "obviously reasonable"
        heuristic -- reinforce with the strongest members first) will erect
        upper-level joints before their only connection to the ground exists,
        which is an immediate self-weight collapse. A build order that always
        finishes connecting each new joint to the already-erected, anchor-
        connected frontier before reaching further out never collapses.

Output (stdin format):
  line 1: W H M D
  next M lines: u v cap      (0-indexed node ids; member index = line order)
  last line: D deck node ids
"""
import sys, random

SEED_BASE = 20260713


def node_id(x, y, W):
    return y * (W + 1) + x


def build_instance(W, H, trap, rng):
    edges = []  # (u, v, cap)
    # risers: level y -> y+1
    for y in range(0, H):
        for x in range(0, W + 1):
            for dx in (-1, 0, 1):
                nx = x + dx
                if 0 <= nx <= W:
                    if trap and y == 0:
                        cap = rng.randint(4, 6)
                    elif trap:
                        cap = rng.randint(7, 9)
                    else:
                        cap = rng.randint(2, 9)
                    edges.append((node_id(x, y, W), node_id(nx, y + 1, W), cap))
    # horizontal braces at levels y=1..H (only when H>=2, keeps single-level
    # cases trivially ordering-safe since every candidate member then touches
    # an anchor directly)
    if H >= 2:
        for y in range(1, H + 1):
            for x in range(0, W):
                if trap:
                    cap = rng.randint(7, 9)
                else:
                    cap = rng.randint(2, 9)
                edges.append((node_id(x, y, W), node_id(x + 1, y, W), cap))
    # deck nodes: spread across x at height H
    xs = sorted(set([max(0, W // 4), W // 2, min(W, (3 * W) // 4)]))
    deck = [node_id(x, H, W) for x in xs]
    return edges, deck


def main():
    test_id = int(sys.argv[1])
    rng = random.Random(SEED_BASE + test_id)

    plan = {
        1: (2, 1, False), 2: (3, 1, False),
        3: (3, 2, False), 4: (4, 2, False), 5: (3, 2, False),
        6: (4, 2, True), 7: (4, 3, True), 8: (5, 3, True),
        9: (5, 3, True), 10: (6, 3, True),
    }
    W, H, trap = plan[min(test_id, 10)]

    edges, deck = build_instance(W, H, trap, rng)
    M = len(edges)
    D = len(deck)

    out = []
    out.append(f"{W} {H} {M} {D}")
    for (u, v, cap) in edges:
        out.append(f"{u} {v} {cap}")
    out.append(" ".join(str(d) for d in deck))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
