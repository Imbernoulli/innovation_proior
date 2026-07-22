#!/usr/bin/env python3
"""Checker for fsx_B_1060 -- Overhang-Supported Additive Print.

CLI: python3 verify.py <in> <out> <ans>   (ans is unused)

Feasibility (any violation -> "Ratio: 0.0"):
  - output is well-formed: "K" then K lines "x y z t" with t in {P,S}, all ints in range,
    no duplicate coordinates, no non-finite/garbage tokens.
  - the P-set (printed part material) equals the target set T EXACTLY.
  - layer-monotone overhang-support: every placed voxel (P or S) at layer z>0 must have at
    least one of {itself, x-1, x+1, y-1, y+1} (the same coordinates, layer z-1) already
    placed (P or S) at layer z-1. z==0 always rests on the build plate (always OK).

Objective (minimize): total support cost.
  - a support (S) voxel costs 1, UNLESS it is "trapped": a part (P) voxel sits directly
    above it anywhere in the same (x,y) column (z2 > z with (x,y,z2) in T) -- trapped
    support material cannot be withdrawn after the print finishes and costs TRAP_MULT (=3).

Baseline B (checker-built, used both for scoring and as the "trivial" reference tier):
  fill the ENTIRE axis-aligned bounding box of T, from z=0 up to max(T.z), as filled
  material (T stays P, everything else in the box becomes S). This is a safe, always-
  feasible, but wasteful construction.

Score: minimization -> sc = min(1000, 100*B/max(1e-9,F)); print "Ratio: %.6f" % (sc/1000).
"""
import sys

TRAP_MULT = 3
MAX_TOKENS = 4_000_000  # hard cap so adversarial "huge" outputs can't blow up memory/time


def fail(msg):
    print("INFEASIBLE:", msg)
    print("Ratio: 0.0")
    sys.exit(0)


def read_ints_line(line):
    parts = line.split()
    out = []
    for p in parts:
        try:
            v = int(p)
        except ValueError:
            return None
        out.append(v)
    return out


def main():
    if len(sys.argv) < 3:
        fail("bad invocation")
    inf, outf = sys.argv[1], sys.argv[2]

    # ---- parse instance ----
    with open(inf, "r") as f:
        in_lines = f.read().split("\n")
    hdr = read_ints_line(in_lines[0])
    if hdr is None or len(hdr) != 3:
        fail("bad instance header")
    Lx, Ly, Lz = hdr
    if not (1 <= Lx <= 100 and 1 <= Ly <= 100 and 1 <= Lz <= 100):
        fail("instance dims out of range")
    tcount = read_ints_line(in_lines[1])
    if tcount is None or len(tcount) != 1:
        fail("bad instance target count")
    T = tcount[0]
    if T <= 0 or T > Lx * Ly * Lz:
        fail("bad instance target count")
    Tset = set()
    for i in range(T):
        row = read_ints_line(in_lines[2 + i])
        if row is None or len(row) != 3:
            fail("bad instance target row")
        x, y, z = row
        if not (0 <= x < Lx and 0 <= y < Ly and 0 <= z < Lz):
            fail("instance target out of bounds")
        Tset.add((x, y, z))
    if len(Tset) != T:
        fail("instance has duplicate target voxels")

    # ---- parse participant output ----
    try:
        with open(outf, "r") as f:
            out_text = f.read()
    except Exception:
        fail("cannot read output")
    toks = out_text.split()
    if not toks:
        fail("empty output")
    if len(toks) > MAX_TOKENS:
        fail("output too large")
    try:
        K = int(toks[0])
    except ValueError:
        fail("bad K")
    if K < 0 or K > 4 * Lx * Ly * Lz + 5:
        fail("K out of plausible range")
    need_toks = 1 + 4 * K
    if len(toks) < need_toks:
        fail("output truncated")
    if len(toks) > need_toks:
        fail("trailing garbage in output")

    Pset = set()
    Sset = set()
    seen = set()
    idx = 1
    for _ in range(K):
        try:
            x = int(toks[idx]); y = int(toks[idx + 1]); z = int(toks[idx + 2])
        except ValueError:
            fail("non-integer coordinate")
        t = toks[idx + 3]
        idx += 4
        if not (0 <= x < Lx and 0 <= y < Ly and 0 <= z < Lz):
            fail("voxel out of bounds")
        coord = (x, y, z)
        if coord in seen:
            fail("duplicate voxel coordinate")
        seen.add(coord)
        if t == "P":
            Pset.add(coord)
        elif t == "S":
            Sset.add(coord)
        else:
            fail("bad voxel type (must be P or S)")

    # ---- P must equal T exactly ----
    if Pset != Tset:
        fail("printed part voxels do not equal the target shape exactly")

    filled = Pset | Sset  # == seen, since no duplicates

    # ---- layer-monotone overhang-support feasibility ----
    for (x, y, z) in filled:
        if z == 0:
            continue
        supported = (
            (x, y, z - 1) in filled or
            (x - 1, y, z - 1) in filled or
            (x + 1, y, z - 1) in filled or
            (x, y - 1, z - 1) in filled or
            (x, y + 1, z - 1) in filled
        )
        if not supported:
            fail(f"unsupported overhang voxel at {(x, y, z)}: layer {z-1} gives it no foothold")

    # ---- objective: support cost with trapped-material penalty ----
    colmaxT = {}
    for (x, y, z) in Tset:
        k = (x, y)
        if k not in colmaxT or z > colmaxT[k]:
            colmaxT[k] = z

    def cost_of(x, y, z):
        cm = colmaxT.get((x, y), -1)
        return TRAP_MULT if cm > z else 1

    F = 0
    for (x, y, z) in Sset:
        F += cost_of(x, y, z)

    # ---- checker's own trivial-but-feasible baseline B: solid bounding box ----
    xs = [c[0] for c in Tset]; ys = [c[1] for c in Tset]; zs = [c[2] for c in Tset]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    zmax = max(zs)
    B = 0
    for bx in range(minx, maxx + 1):
        for by in range(miny, maxy + 1):
            for bz in range(0, zmax + 1):
                if (bx, by, bz) not in Tset:
                    B += cost_of(bx, by, bz)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("support_cost F=%d baseline B=%d" % (F, B))
    print("Ratio: %.6f" % (sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
