# TIER: greedy
#!/usr/bin/env python3
"""Recipe approach: every round, scan lanes in the order they were declared and
admit up to Ccap ready moves, first-come-first-served -- the natural "just simulate
everyone, round by round" first attempt. It never starves any lane outright and
never produces an infeasible program, but a fixed scan order has no notion of which
lane is *urgent*: if many short (single-move) lanes are declared before a long
multi-round lane, this scheduler drains all of them before the long lane ever gets
a slot, so the long lane's already-unavoidable critical path only starts once
every short lane is already finished -- wasting rounds that a priority aware
scheduler would have hidden behind the long lane's latency."""
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
    groups = defaultdict(list)
    for i in range(N):
        sx, sy, sz = starts[i]
        groups[(sy, sz)].append(i)
    lanes = []
    for key in sorted(groups.keys()):  # declaration order == (y,z) row order
        ids = groups[key]
        ids.sort(key=lambda i: -starts[i][0])
        lanes.append(ids)
    return lanes


def schedule(lanes, Ccap):
    """Fixed-priority (declaration-order) admission control, Ccap moves/round."""
    cursor = [0] * len(lanes)
    blocks = []
    remaining = sum(len(l) for l in lanes)
    while remaining > 0:
        blk = []
        for li, lane in enumerate(lanes):
            if len(blk) >= Ccap:
                break
            if cursor[li] < len(lane):
                blk.append(lane[cursor[li]])
                cursor[li] += 1
                remaining -= 1
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
