#!/usr/bin/env python3
"""
fsx_S_0889 -- Hot-Loop Scheduler for an In-Order Dual-Issue Core.

Builds K independent micro-op "accumulate chains" (streams). Stream s is a
straight-line dependency chain of L rounds; round r is
    L_r: load addr_s
    A_r: alu  (depends on L_r and on A_{r-1})
    S_r: store addr_s (depends on A_r)
    (S_r must precede L_{r+1}: loop-carried WAR/WAW hazard on addr_s)
so each stream is a single, strictly-ordered dependency chain of 3*L nodes.
addr_s is chosen so that addr_s % S == s % S: streams are deliberately
partitioned into S "cache-set classes"; several streams alias the SAME
class whenever K > S. In addition F independent single-node ALU "filler"
ops (no address, no dependency) are emitted -- unrelated basic-block work
available to fill an idle issue slot.

Node ids are relabelled with a random bijection (seeded by testId) and both
the node-definition lines and the edge lines are re-sorted by the new label,
so the input's line order carries no structural hint: a solver must read
the dependence edges and the addresses to recover which nodes belong to the
same chain and which chains alias the same cache set.
"""
import sys
import random

ALU, LOAD, STORE = "A", "L", "S"

# (K streams, L rounds/stream, S cache sets, F filler ops) per testId.
TESTS = [
    (3, 3, 3, 2),    # 1  warm-up: one stream per set, no aliasing
    (4, 3, 4, 3),    # 2  warm-up: one stream per set, no aliasing
    (4, 4, 2, 4),    # 3  trap begins: 2 streams alias each set
    (6, 4, 2, 4),    # 4  trap: 3 streams alias each set
    (6, 5, 3, 6),    # 5  trap: 2 streams alias each set
    (8, 5, 2, 6),    # 6  trap: 4 streams alias each set
    (8, 6, 4, 8),    # 7  trap: 2 streams alias each set
    (10, 6, 2, 8),   # 8  trap: 5 streams alias each set
    (10, 7, 3, 10),  # 9  trap: 3-4 streams alias each set
    (12, 8, 3, 12),  # 10 largest / hardest: 4 streams alias each set
]


def main():
    tid = int(sys.argv[1])
    K, L, S, F = TESTS[(tid - 1) % len(TESTS)]
    rng = random.Random(7_654_321 * tid + 991)

    nodes = []   # list of (type, addr_or_None)
    edges = []   # (u_internal, v_internal), 0-indexed

    def add_node(t, a=None):
        nodes.append((t, a))
        return len(nodes) - 1

    for s in range(K):
        addr = (s % S) + S * (5000 + 37 * s)
        prev_store = None
        prev_alu = None
        for r in range(L):
            lid = add_node(LOAD, addr)
            if prev_store is not None:
                edges.append((prev_store, lid))
            aid = add_node(ALU)
            edges.append((lid, aid))
            if prev_alu is not None:
                edges.append((prev_alu, aid))
            sid = add_node(STORE, addr)
            edges.append((aid, sid))
            prev_store, prev_alu = sid, aid

    for _ in range(F):
        add_node(ALU)

    N = len(nodes)
    M = len(edges)

    perm = list(range(N))
    rng.shuffle(perm)  # perm[internal_id] = external label

    lines = []
    for i, (t, a) in enumerate(nodes):
        label = perm[i]
        if t == ALU:
            lines.append((label, f"{label} A"))
        else:
            lines.append((label, f"{label} {t} {a}"))
    lines.sort(key=lambda x: x[0])

    edge_lines = sorted((perm[u], perm[v]) for u, v in edges)

    out = [f"{N} {M} {S}"]
    out.extend(text for _, text in lines)
    out.extend(f"{u} {v}" for u, v in edge_lines)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
