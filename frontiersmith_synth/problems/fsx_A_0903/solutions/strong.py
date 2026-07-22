# TIER: strong
#!/usr/bin/env python3
"""The insight: build the collision-dependence graph over ALL drones from the raw
coordinates (two drones are dependent iff they could ever occupy the same cell --
here, iff they share a (y,z) row). Lanes sit on disjoint rows, so the graph has NO
edges between lanes: the fleet decomposes into |lanes| independent chains, each
already at its own irreducible critical-path length (the single-gap cascade). The
only shared bottleneck left is the uplink cap Ccap. Under a shared capacity, a lane
whose remaining chain is LONG should never be starved -- every round it is ready it
should be admitted first, because its remaining length lower-bounds the whole
program's makespan regardless of what else runs; short (already-critical-path-free)
lanes should fill whatever capacity is left over. This "longest remaining chain
first" admission keeps the long lane's inherent critical path fully hidden behind
the short lanes' work instead of paying for it twice."""
import sys
from collections import defaultdict


def read_instance():
    toks = sys.stdin.read().split()
    it = iter(toks)
    X = int(next(it)); Y = int(next(it)); Z = int(next(it))
    N = int(next(it)); Bc = int(next(it)); Ccap = int(next(it))
    starts, goals = [], []
    for _ in range(N):
        sx, sy, sz, gx, gy, gz = (int(next(it)) for _ in range(6))
        starts.append((sx, sy, sz))
        goals.append((gx, gy, gz))
    return X, Y, Z, N, Bc, Ccap, starts, goals


def dir_token(s, g):
    dx, dy, dz = g[0] - s[0], g[1] - s[1], g[2] - s[2]
    if dx == 1: return "PX"
    if dx == -1: return "NX"
    if dy == 1: return "PY"
    if dy == -1: return "NY"
    if dz == 1: return "PZ"
    if dz == -1: return "NZ"
    raise ValueError("unsupported single-cell delta %r" % ((dx, dy, dz),))


def build_lanes(starts, N):
    """The 'collision-dependence graph': group drones sharing a (y,z) row (the
    only coordinate combination that can ever collide, since motion here never
    changes y or z). Different groups are provably disjoint components."""
    groups = defaultdict(list)
    for i in range(N):
        sx, sy, sz = starts[i]
        groups[(sy, sz)].append(i)
    lanes = []
    for key in sorted(groups.keys()):
        ids = groups[key]
        ids.sort(key=lambda i: -starts[i][0])
        lanes.append(ids)
    return lanes


def schedule(lanes, Ccap):
    """Longest-remaining-chain-first admission control, Ccap moves/round."""
    cursor = [0] * len(lanes)
    blocks = []
    remaining_total = sum(len(l) for l in lanes)
    while remaining_total > 0:
        remaining_len = [len(l) - cursor[i] for i, l in enumerate(lanes)]
        order = sorted(
            (i for i in range(len(lanes)) if remaining_len[i] > 0),
            key=lambda i: (-remaining_len[i], i),
        )
        blk = []
        for li in order:
            if len(blk) >= Ccap:
                break
            lane = lanes[li]
            blk.append(lane[cursor[li]])
            cursor[li] += 1
            remaining_total -= 1
        blocks.append(blk)
    return blocks


def main():
    X, Y, Z, N, Bc, Ccap, starts, goals = read_instance()
    lanes = build_lanes(starts, N)
    blocks = schedule(lanes, Ccap)

    lines = []
    for bi, blk in enumerate(blocks):
        for d in blk:
            lines.append(f"MOVE {d} {dir_token(starts[d], goals[d])}")
        if bi != len(blocks) - 1:
            lines.append("BARRIER")
    out = [str(len(lines))] + lines
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
