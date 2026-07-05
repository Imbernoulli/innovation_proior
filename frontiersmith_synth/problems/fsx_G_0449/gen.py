#!/usr/bin/env python3
"""Generate ONE "graphics-pipeline composition" instance (format D).

python3 gen.py <testId>   -> prints one instance to stdout.
testId 1..10 is a difficulty ladder (longer chains, larger / more irregular
dimensions, richer stage-type mix).  Randomness is seeded ONLY by testId, so
the instance is fully deterministic.

STORY
-----
A rendering pipeline composes a chain of linear transforms

    T = S_1 . S_2 . ... . S_L        (matrix product, applied left-to-right)

Each stage S_i maps an attribute space of dimension d_i to one of dimension
d_{i+1} (positions, normals, tangents, uv, blend weights, ... all have
different widths, so the dimensions are IRREGULAR).  A stage is given in one
of three authentic forms:

  DENSE   : an explicit d_i x d_{i+1} transform matrix D.
  LOWRANK : a rank-r deformation given only as its factors  U (d_i x r) and
            V (r x d_{i+1});  the stage matrix is  U . V  (never materialized
            for you).
  SUMLR   : a base transform plus a low-rank correction:  A + B . C  with
            A (d_i x d_{i+1}), B (d_i x r), C (r x d_{i+1}).

Your job: emit a straight-line program of matrix operations (multiplies and
adds) that computes the composite matrix T using AS FEW SCALAR MULTIPLIES as
possible.  You may reorder the product (associativity), keep low-rank factors
un-multiplied, and distribute sums (distributivity) however you like -- any
circuit that reproduces T exactly is accepted; only the multiply count scores.

INSTANCE SCHEMA (stdin for solver / <in> for checker)
-----------------------------------------------------
  line 1:  L
  line 2:  d_0 d_1 ... d_L
  then L stage blocks, block i one of:
     DENSE
     <d_i lines of d_{i+1} ints>            # D
   |
     LOWRANK r
     <d_i lines of r ints>                  # U
     <r lines of d_{i+1} ints>              # V
   |
     SUMLR r
     <d_i lines of d_{i+1} ints>            # A
     <d_i lines of r ints>                  # B
     <r lines of d_{i+1} ints>              # C

The GIVEN matrices are numbered 0,1,2,... in the exact order they appear in
the stage blocks (DENSE -> [D]; LOWRANK -> [U,V]; SUMLR -> [A,B,C]).  These
indices are the leaf ids your straight-line program refers to.
"""
import sys
import random


def emit_matrix(rng, r, c, out):
    for _ in range(r):
        row = [str(rng.randint(-3, 3)) for _ in range(c)]
        out.append(" ".join(row))


def main():
    tid = int(sys.argv[1])
    if tid < 1:
        tid = 1
    rng = random.Random(44900 + tid * 6151)

    # ---- difficulty ladder ----
    L = 3 + tid                      # chain length 4 .. 13
    dim_pool_small = [4, 5, 6, 8]
    dim_pool_big = [4, 6, 8, 10, 12, 14, 16, 18, 20, 24]
    pool = dim_pool_small if tid <= 3 else dim_pool_big

    # irregular dimensions (non-monotone => reordering genuinely helps)
    dims = [rng.choice(pool) for _ in range(L + 1)]

    # stage-type mix: bias LARGER instances toward SUMLR/DENSE so the low-rank
    # advantage stays bounded (keeps headroom under the 10x score cap), while
    # keeping enough LOWRANK stages for strategy divergence.
    if tid <= 3:
        type_weights = [("LOWRANK", 5), ("DENSE", 3), ("SUMLR", 2)]
    elif tid <= 6:
        type_weights = [("LOWRANK", 4), ("DENSE", 3), ("SUMLR", 3)]
    else:
        type_weights = [("LOWRANK", 3), ("DENSE", 3), ("SUMLR", 4)]
    bag = []
    for name, w in type_weights:
        bag += [name] * w

    lines = []
    lines.append(str(L))
    lines.append(" ".join(str(d) for d in dims))

    have_lowrank = False
    for i in range(L):
        din, dout = dims[i], dims[i + 1]
        typ = rng.choice(bag)
        # force at least one LOWRANK so factoring strategy always applies
        if i == L - 1 and not have_lowrank:
            typ = "LOWRANK"
        mind = min(din, dout)
        # moderate rank: >= 2, roughly 40-60% of the smaller side, and strictly
        # smaller than the smaller side so factoring can actually save work.
        rmax = max(2, min(mind - 1, (mind + 1) // 2 + 1))
        r = rng.randint(2, rmax) if rmax >= 2 else 2
        if r >= mind:
            r = max(2, mind - 1)

        if typ == "DENSE":
            lines.append("DENSE")
            emit_matrix(rng, din, dout, lines)
        elif typ == "LOWRANK":
            have_lowrank = True
            lines.append("LOWRANK %d" % r)
            emit_matrix(rng, din, r, lines)     # U
            emit_matrix(rng, r, dout, lines)    # V
        else:  # SUMLR
            lines.append("SUMLR %d" % r)
            emit_matrix(rng, din, dout, lines)  # A
            emit_matrix(rng, din, r, lines)     # B
            emit_matrix(rng, r, dout, lines)    # C

    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
