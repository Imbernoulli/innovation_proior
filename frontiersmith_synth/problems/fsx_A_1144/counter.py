#!/usr/bin/env python3
"""
counter.py <in> <out> <ans>   (ans is an unused placeholder)

Format D (op-count) checker for "build the sculpture in fewest moves".

The instance (<in>) is a target voxel set (a list of integer (x,y,z) points).
The participant (<out>) prints a STRAIGHT-LINE PROGRAM over four fixed
growth macros, one op per line, assemblies numbered 0,1,2,... in the order
they are created:

    U                creates {(0,0,0)}                         (place a unit)
    T i dx dy dz      creates assembly i translated by (dx,dy,dz)
    R i a             creates assembly i reflected through the plane
                       perpendicular to axis a in {0,1,2} (negate that coord)
    M i j             creates union(assembly i, assembly j)

i, j must reference an EARLIER assembly (no forward/self references at
creation time). The FINAL assembly (the one created by the last op) must
equal the target voxel set EXACTLY (checked as an exact set equality, no
tolerance -- this is discrete geometry). The objective is the number of op
lines (fewer is better).

Scoring: an ops-count is normalized on a LOG scale between two internally
computed reference points -- a naive per-voxel baseline B_hi = 2V-1 (one U
per voxel + a union tree) that ANY correct program can trivially achieve
(-> ratio 0.10), and an information-theoretic floor B_lo = ceil(log2 V) + 1
(a hard lower bound: each op at best DOUBLES the size of the largest
assembly built so far, so reaching size V needs >= log2(V)+1 ops in total --
no correct program can ever beat B_lo, so it is a safe, honest reference
that never saturates the score). Score = 0.1 + 0.7 * clamp(frac,0,1) where
frac = (log(B_hi) - log(F)) / (log(B_hi) - log(B_lo)); max attainable score
is 0.8, always leaving headroom.

Hardened: bounded op count, bounded per-op coordinate magnitudes, and a
cumulative "work budget" (sum of assembly sizes touched) so that a submitted
program cannot use the very doubling trick this problem rewards to blow up
checker memory/time -- any breach is scored as infeasible (Ratio 0.0).
"""
import sys
import math

MAX_OPS = 300000          # generous vs. worst-case legit trivial baseline (2V-1)
MAX_COORD = 10 ** 7        # sanity bound on any single coordinate / translate delta
WORK_BUDGET = 5_000_000    # total "points touched" across the whole replay


def fail(reason):
    print("infeasible: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def read_instance(path):
    try:
        with open(path) as f:
            toks = f.read().split()
    except Exception:
        fail("cannot read instance")
    if not toks:
        fail("empty instance")
    try:
        v = int(toks[0])
    except Exception:
        fail("bad instance header")
    if v <= 0 or len(toks) != 1 + 3 * v:
        fail("bad instance shape")
    try:
        vals = [int(t) for t in toks[1:]]
    except Exception:
        fail("bad instance coordinates")
    pts = set()
    for k in range(v):
        pts.add((vals[3 * k], vals[3 * k + 1], vals[3 * k + 2]))
    if len(pts) != v:
        fail("instance has duplicate points")  # shouldn't happen, defensive
    return pts


def parse_int_tok(tok):
    if len(tok) > 32:
        fail("token too long")
    low = tok.lower()
    if "nan" in low or "inf" in low:
        fail("non-finite token")
    try:
        return int(tok)
    except Exception:
        fail("unparsable integer token '%s'" % tok[:32])


def main():
    if len(sys.argv) < 3:
        fail("usage")
    in_path, out_path = sys.argv[1], sys.argv[2]
    target = read_instance(in_path)
    V = len(target)

    try:
        with open(out_path) as f:
            lines = f.readlines()
    except Exception:
        fail("cannot read output")

    assemblies = []   # list of frozensets (assembly index -> point set)
    ops = 0
    work = 0

    def check_idx(tok):
        idx = parse_int_tok(tok)
        if idx < 0 or idx >= len(assemblies):
            fail("assembly index out of range: %s" % tok)
        return idx

    for raw in lines:
        parts = raw.split()
        if not parts:
            continue
        ops += 1
        if ops > MAX_OPS:
            fail("program too long (> %d ops)" % MAX_OPS)
        op = parts[0].upper()

        if op == "U":
            if len(parts) != 1:
                fail("U takes no arguments")
            new = frozenset({(0, 0, 0)})
            work += 1

        elif op == "T":
            if len(parts) != 5:
                fail("T arity")
            i = check_idx(parts[1])
            dx = parse_int_tok(parts[2])
            dy = parse_int_tok(parts[3])
            dz = parse_int_tok(parts[4])
            if abs(dx) > MAX_COORD or abs(dy) > MAX_COORD or abs(dz) > MAX_COORD:
                fail("translate delta out of range")
            src = assemblies[i]
            work += len(src)
            if work > WORK_BUDGET:
                fail("resource limit exceeded (work budget)")
            new = frozenset((x + dx, y + dy, z + dz) for (x, y, z) in src)

        elif op == "R":
            if len(parts) != 3:
                fail("R arity")
            i = check_idx(parts[1])
            a = parse_int_tok(parts[2])
            if a not in (0, 1, 2):
                fail("reflect axis must be 0,1,2")
            src = assemblies[i]
            work += len(src)
            if work > WORK_BUDGET:
                fail("resource limit exceeded (work budget)")
            if a == 0:
                new = frozenset((-x, y, z) for (x, y, z) in src)
            elif a == 1:
                new = frozenset((x, -y, z) for (x, y, z) in src)
            else:
                new = frozenset((x, y, -z) for (x, y, z) in src)

        elif op == "M":
            if len(parts) != 3:
                fail("M arity")
            i = check_idx(parts[1])
            j = check_idx(parts[2])
            a, b = assemblies[i], assemblies[j]
            work += len(a) + len(b)
            if work > WORK_BUDGET:
                fail("resource limit exceeded (work budget)")
            new = a | b

        else:
            fail("unknown op '%s'" % op[:16])

        if len(new) > 4 * V + 16:
            fail("intermediate assembly grew far beyond target size")

        assemblies.append(new)
        if len(assemblies) > MAX_OPS + 4:
            fail("too many assemblies")

    if ops == 0 or not assemblies:
        fail("empty program")

    final = assemblies[-1]
    if final != target:
        missing = len(target - final)
        extra = len(final - target)
        fail("final assembly != target (missing=%d extra=%d)" % (missing, extra))

    F = ops
    B_hi = 2 * V - 1
    B_lo = math.ceil(math.log2(V)) + 1 if V > 1 else 1
    if B_hi <= B_lo:
        B_hi = B_lo + 1
    if F < B_lo:
        F = B_lo  # cannot happen for a correct program (proven lower bound); guard only
    lo, hi = math.log(B_lo), math.log(B_hi)
    frac = (hi - math.log(F)) / (hi - lo)
    if frac < 0.0:
        frac = 0.0
    if frac > 1.0:
        frac = 1.0
    score = 0.1 + 0.7 * frac
    if score < 0.0:
        score = 0.0
    if score > 1.0:
        score = 1.0

    print("V=%d F=%d B_hi=%d B_lo=%d" % (V, F, B_hi, B_lo))
    print("Ratio: %.6f" % score)
    sys.exit(0)


if __name__ == "__main__":
    main()
