#!/usr/bin/env python3
"""gen.py <testId> -- DroneScript Barrier Compiler instance generator.

Plants K "conveyor lanes": lane L is a straight corridor of d_L+1 cells along the
x-axis, occupying a unique (y,z) row so that DIFFERENT lanes never share a cell no
matter what a solver does with them -- spatial collisions never happen across lanes.
Within lane L, d_L drones sit at x=0..d_L-1 and must each shift one cell to the right
(goal x = start x + 1); the corridor's last cell (x=d_L) starts empty. Because a
block may only move a drone into a cell that is UNOCCUPIED before the block (see
statement.md), a single lane's shift is an unavoidable d_L-round dependency chain.

On top of that, EVERY block obeys a global uplink cap `Ccap`: at most Ccap MOVE
instructions total, across ALL lanes, may appear in one block (the ground station
can only issue Ccap simultaneous fly commands per synchronization tick). Instances
mix MANY short (d=1) lanes with one or two LONG lanes. A scheduler that always
admits whichever lanes it encounters first (the natural fixed-priority read of the
input) drains every short lane before ever giving the long lane a slot, stalling the
long lane's already-latent critical path -- provably more barriers than a scheduler
that recognizes the long lane must be kept moving every round it can.
"""
import sys
import random


def lanes_for(tid):
    # (num_short_lanes, [long_lane_lengths], Ccap, Bc)
    table = {
        1: (3, [3], 3, 5),
        2: (6, [4], 3, 6),
        3: (8, [5], 3, 6),
        4: (12, [6], 3, 7),
        5: (16, [7], 4, 8),
        6: (20, [8], 4, 8),
        7: (28, [9], 4, 9),
        8: (34, [10], 5, 9),
        9: (40, [12], 5, 10),
        10: (48, [14, 6], 5, 10),
    }
    s, longs, C, Bc = table[tid]
    rnd = random.Random(5000 + 13 * tid)
    jitter = 0 if tid <= 3 else 1
    s = max(1, s + (rnd.randint(-jitter, jitter) if jitter else 0))
    longs = [max(2, l + (rnd.randint(-jitter, jitter) if jitter else 0)) for l in longs]
    return s, longs, C, Bc


def main():
    tid = int(sys.argv[1])
    s, longs, Ccap, Bc = lanes_for(tid)
    d = [1] * s + list(longs)          # declaration order: short lanes first, long(s) last
    K = len(d)
    X = max(d) + 1
    Y = K
    Z = 2

    drones = []
    for lane_idx, dl in enumerate(d):
        y = lane_idx
        z = lane_idx % 2
        for t in range(dl):
            sx, gx = t, t + 1
            drones.append((sx, y, z, gx, y, z))
    N = len(drones)

    out = [f"{X} {Y} {Z} {N} {Bc} {Ccap}"]
    for (sx, sy, sz, gx, gy, gz) in drones:
        out.append(f"{sx} {sy} {sz} {gx} {gy} {gz}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
