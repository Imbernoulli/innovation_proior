#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE instance (a 3-D "phasor coupling tensor") to stdout.

POWER-GRID skin (family: tensor-decomposition-rank, format D, AlphaEvolve-inspired):
A substation's real-time state estimator holds a three-way coupling table
    T[i][j][k] = the joint coupling coefficient among bus i, feeder line j and
                 harmonic band k.
The controller evaluates T as a sum of rank-1 "multiplier primitives"
    (a_r . busVec)(b_r . lineVec)(c_r . harmVec),
and each primitive burns exactly ONE hardware scalar multiply. Fewer primitives =
cheaper controller. We ship a coupling tensor whose HARMONIC-BAND slices (one matrix
per band k) are each planted as a LOW-RANK bus x line matrix, yet the total number of
planted primitives EXCEEDS every tensor dimension (over-complete rank). Consequently no
polynomial diagonalization (Jennrich / simultaneous-diagonalization needs rank <= a mode
dimension) can recover the true minimum -- the genuine optimum stays unknown.

Deterministic: everything is seeded by testId only. Integer entries.

STDOUT format:
    B L H                       (buses, feeder lines, harmonic bands)
    then H harmonic-band slices; slice k is B lines of L integers  ( = T[i][j][k]).
"""
import sys
import random

# Row = (B, L, H, s):
#   H  = number of harmonic bands (smallest mode -> fewest slices)
#   s  = planted matrix rank of EACH band slice
# Invariants that make the instance well-posed & hard:
#   s < min(B, L)     -> slice factoring strictly beats every fiber decomposition
#   s*H > max(B, L)   -> total planted rank is OVER-COMPLETE (diagonalization fails)
PARAMS = {
    1:  (4, 4, 3, 2),
    2:  (4, 5, 3, 2),
    3:  (5, 5, 3, 2),
    4:  (5, 5, 2, 3),
    5:  (5, 6, 4, 2),
    6:  (6, 6, 4, 2),
    7:  (6, 6, 3, 3),
    8:  (6, 7, 4, 2),
    9:  (7, 7, 4, 2),
    10: (7, 7, 3, 3),
}


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if t not in PARAMS:
        t = ((t - 1) % len(PARAMS)) + 1
    B, L, H, s = PARAMS[t]
    rng = random.Random(60050 + 7919 * t)

    T = [[[0] * H for _ in range(L)] for _ in range(B)]

    def nonzero_vec(n):
        while True:
            v = [rng.randint(-4, 4) for _ in range(n)]
            if any(x != 0 for x in v):
                return v

    # Build each harmonic-band slice as the sum of s integer rank-1 outer products
    # busPattern (x) linePattern. Generic rank of the slice = s; planted total = s*H.
    for k in range(H):
        for _ in range(s):
            bus = nonzero_vec(B)
            line = nonzero_vec(L)
            for i in range(B):
                if bus[i] == 0:
                    continue
                bi = bus[i]
                for j in range(L):
                    T[i][j][k] += bi * line[j]

    out = ["%d %d %d" % (B, L, H)]
    for k in range(H):
        for i in range(B):
            out.append(" ".join(str(T[i][j][k]) for j in range(L)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
