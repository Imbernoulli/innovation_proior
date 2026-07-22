# TIER: trivial
#!/usr/bin/env python3
"""Naive baseline: route every drone independently, one MOVE followed by its own
BARRIER, never batching two drones into the same block. Always safe (a block with
one MOVE trivially respects any Ccap>=1) but pays a barrier for every single move."""
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
    """Group drones sharing a (y,z) row into a lane; within a lane the drone
    closest to the empty slot (largest x) must move first."""
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


def main():
    X, Y, Z, N, Bc, Ccap, starts, goals = read_instance()
    lanes = build_lanes(starts, N)
    moves = []
    for lane in lanes:
        for d in lane:
            moves.append((d, dir_token(starts[d], goals[d])))
    lines = []
    for idx, (d, dt) in enumerate(moves):
        lines.append(f"MOVE {d} {dt}")
        if idx != len(moves) - 1:
            lines.append("BARRIER")
    out = [str(len(lines))] + lines
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
