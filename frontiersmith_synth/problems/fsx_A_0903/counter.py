#!/usr/bin/env python3
"""counter.py <in> <out> <ans> -- DroneScript Barrier Compiler checker.

Feasibility (strict, in order):
  - output must start with an integer P (0 <= P <= MAX_P) followed by exactly P
    instruction lines: "MOVE d DIR", "HOLD d", "BARRIER".
  - within one block (the run of instructions between BARRIERs / program ends),
    a drone may receive at most one instruction; a MOVE's destination must be
    in-bounds AND must not be occupied, at the START of the block, by ANY drone
    (moving or not) -- this bars both "flying into a parked drone" and "domino
    chains" through a drone that is itself leaving that same block; and no two
    movers in the same block may target the same destination cell (this also
    subsumes direct swap conflicts: if A's target is B's pre-block cell, B is
    occupying it at block start, so it is already rejected).
  - a block may contain at most Ccap MOVE instructions in total (the shared
    uplink budget), regardless of how many different lanes/drones are involved.
  - at program end every drone must be exactly on its goal cell.
Any violation -> "Ratio: 0.0". Otherwise score = min(1, 0.1 * Base / F) where
F = (#MOVE) + Bc*(#BARRIER) and Base = Bc-inflated per-drone-Manhattan baseline
(move every drone alone, one MOVE then one BARRIER, never batching anything).
"""
import re
import sys

MAX_P = 2_000_000
INT_RE = re.compile(r"^[+-]?\d+$")
DIRS = {
    "PX": (1, 0, 0), "NX": (-1, 0, 0),
    "PY": (0, 1, 0), "NY": (0, -1, 0),
    "PZ": (0, 0, 1), "NZ": (0, 0, -1),
}


def strict_int(tok):
    if tok is None or not INT_RE.match(tok):
        return None
    try:
        return int(tok)
    except ValueError:
        return None


def fail(reason):
    print("infeasible: " + reason)
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    if len(sys.argv) < 3:
        fail("bad invocation")
    inf, outf = sys.argv[1], sys.argv[2]

    in_toks = open(inf).read().split()
    pos_i = [0]

    def nxt():
        v = in_toks[pos_i[0]]
        pos_i[0] += 1
        return v

    X, Y, Z, N, Bc, Ccap = (int(nxt()) for _ in range(6))
    starts, goals = [], []
    for _ in range(N):
        sx, sy, sz, gx, gy, gz = (int(nxt()) for _ in range(6))
        starts.append((sx, sy, sz))
        goals.append((gx, gy, gz))

    try:
        raw = open(outf, "r", errors="replace").read()
    except Exception:
        fail("cannot read output")

    lines = raw.split("\n")
    if not lines or lines[0].strip() == "":
        fail("missing P")
    Pv = strict_int(lines[0].strip())
    if Pv is None:
        fail("P is not an integer token")
    P = Pv
    if P < 0 or P > MAX_P:
        fail("P out of range")
    if len(lines) - 1 < P:
        fail("fewer than P instruction lines present")
    instr_lines = lines[1:1 + P]
    for extra in lines[1 + P:]:
        if extra.strip() != "":
            fail("trailing non-blank content after the declared P instruction lines")

    pos = list(starts)
    total_moves = 0
    total_barriers = 0
    block_moved = {}   # drone -> new position, pending this block
    block_touched = set()

    def commit_block():
        nonlocal pos
        if not block_moved:
            return True
        old_occupied = set(pos)
        dests = list(block_moved.values())
        if len(set(dests)) != len(dests):
            return False
        for dst in dests:
            if dst in old_occupied:
                return False
        newpos = list(pos)
        for dr, np_ in block_moved.items():
            newpos[dr] = np_
        pos = newpos
        return True

    for raw_line in instr_lines:
        toks = raw_line.split()
        if not toks:
            fail("empty instruction line")
        op = toks[0]
        if op == "BARRIER":
            if len(toks) != 1:
                fail("malformed BARRIER line")
            if not commit_block():
                fail("collision inside a block")
            block_moved = {}
            block_touched = set()
            total_barriers += 1
        elif op == "MOVE":
            if len(toks) != 3:
                fail("malformed MOVE line")
            d = strict_int(toks[1])
            if d is None or not (0 <= d < N):
                fail("bad drone id in MOVE")
            if toks[2] not in DIRS:
                fail("bad direction token")
            if d in block_touched:
                fail("drone double-scheduled inside one block")
            if len(block_moved) >= Ccap:
                fail("block exceeds the Ccap concurrent-MOVE uplink budget")
            dx, dy, dz = DIRS[toks[2]]
            px, py, pz = pos[d]
            npos = (px + dx, py + dy, pz + dz)
            if not (0 <= npos[0] < X and 0 <= npos[1] < Y and 0 <= npos[2] < Z):
                fail("MOVE leaves the grid")
            block_moved[d] = npos
            block_touched.add(d)
            total_moves += 1
        elif op == "HOLD":
            if len(toks) != 2:
                fail("malformed HOLD line")
            d = strict_int(toks[1])
            if d is None or not (0 <= d < N):
                fail("bad drone id in HOLD")
            if d in block_touched:
                fail("drone double-scheduled inside one block")
            block_touched.add(d)
        else:
            fail("unknown instruction " + op)

    if not commit_block():
        fail("collision inside the trailing block")

    for i in range(N):
        if pos[i] != goals[i]:
            fail(f"drone {i} did not reach its goal")

    if total_moves == 0 and N > 0:
        fail("no drone ever moved")

    F = total_moves + total_barriers * Bc
    if F <= 0:
        fail("nonpositive program cost")

    M = sum(abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])
            for a, b in zip(starts, goals))
    Base = M * (1 + Bc)

    sc = min(1000.0, 100.0 * Base / max(1e-9, F))
    print(f"F={F} Base={Base} moves={total_moves} barriers={total_barriers}")
    print("Ratio: %.6f" % (sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
