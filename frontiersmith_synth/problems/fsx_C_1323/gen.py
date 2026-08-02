#!/usr/bin/env python3
"""gen.py <testId> -- print ONE scaffold-hop instance to stdout.

A "molecule" is a chain of fragments (0-indexed positions 0..L-1). Fragment at
position i places its single atom at global coordinate (i*STEP+dx, dy, dz).
The KNOWN ACTIVE is one such chain; its activity is pinned by K pharmacophore
anchors (type + 3D point + tolerance), each satisfied by one of its own atoms.

The fragment library is FIXED (same 15 fragments in every instance): a common
filler (cheap, no feature), 4 common feature fragments (one per type D/A/R/H,
expensive), a cheaper "novel" filler, 4 cheaper "novel" feature fragments
(one per type, absent from every known active), a very expensive decoy, and 4
"near miss" feature fragments whose atom sits just OUTSIDE the tolerance.

Only the known active's length/anchor layout varies with testId (a difficulty
ladder). Seeded by testId only -> bit-for-bit reproducible.
"""
import sys
import random

STEP = 1000
TOL = 0.5

# (cost, type, dx, dy, dz) indexed by fragment id 0..14. type in {D,A,R,H,X}.
LIBRARY = [
    (4, 'X', 0, 0, 0),        # 0  filler_common
    (9, 'D', 2, 0, 0),        # 1  feature_common D
    (9, 'A', -2, 0, 0),       # 2  feature_common A
    (9, 'R', 0, 2, 0),        # 3  feature_common R
    (9, 'H', 0, -2, 0),       # 4  feature_common H
    (2, 'X', 0, 0, 0),        # 5  filler_novel   (cheapest overall)
    (6, 'D', 2, 0, 0),        # 6  feature_novel D
    (6, 'A', -2, 0, 0),       # 7  feature_novel A
    (6, 'R', 0, 2, 0),        # 8  feature_novel R
    (6, 'H', 0, -2, 0),       # 9  feature_novel H
    (18, 'X', 0, 0, 0),       # 10 decoy_expensive
    (5, 'D', 2.6, 0, 0),      # 11 near_miss D  (dist 0.6 > TOL from canonical)
    (5, 'A', -2.6, 0, 0),     # 12 near_miss A
    (5, 'R', 0, 2.6, 0),      # 13 near_miss R
    (5, 'H', 0, -2.6, 0),     # 14 near_miss H
]
M = len(LIBRARY)
FEATURE_COMMON = {'D': 1, 'A': 2, 'R': 3, 'H': 4}

# difficulty ladder: (L_ref, K anchors), K/L_ref kept in [0.28,0.38] on purpose
# so the "keep-anchor-identity" trap bites on every case.
SPECS = [
    (6, 2), (7, 2), (8, 3), (9, 3), (10, 3),
    (11, 4), (12, 4), (13, 4), (14, 5), (16, 6),
]
PAD_BUFFER = 8


def main():
    tid = int(sys.argv[1])
    rng = random.Random(20260702 + 9173 * tid)

    L_ref, K = SPECS[(tid - 1) % len(SPECS)]
    L_max = L_ref + PAD_BUFFER

    anchor_idx = sorted(rng.sample(range(L_ref), K))
    seq = [0] * L_ref  # default: filler_common everywhere
    for idx in anchor_idx:
        t = rng.choice(['D', 'A', 'R', 'H'])
        seq[idx] = FEATURE_COMMON[t]

    S_ref = sum(LIBRARY[fid][0] for fid in seq)
    BUDGET = 3 * S_ref

    anchors = []
    for idx in anchor_idx:
        fid = seq[idx]
        cost, typ, dx, dy, dz = LIBRARY[fid]
        x = idx * STEP + dx
        y = dy
        z = dz
        anchors.append((x, y, z, typ, TOL))

    out = []
    out.append("%d %d %d %d" % (M, L_max, STEP, BUDGET))
    for fid in range(M):
        cost, typ, dx, dy, dz = LIBRARY[fid]
        out.append("%d %d %s %s %s %s" % (fid, cost, typ, dx, dy, dz))
    out.append(str(L_ref))
    out.append(" ".join(str(x) for x in seq))
    out.append(str(K))
    for (x, y, z, typ, tol) in anchors:
        out.append("%s %s %s %s %s" % (x, y, z, typ, tol))

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
