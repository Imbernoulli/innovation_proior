#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>  -- deterministic checker for the banquet replica-router.

Artifact (participant stdout) =
  D lines: dish d's R replica-kitchen ids (0-indexed, distinct, in [0,K))
  B lines: for course c (input order), the kitchen id that actually plates each
           dish of that course, in the SAME order the dishes were listed in the
           input for that course; each chosen kitchen must be one of that dish's
           R declared replica kitchens.

Feasibility (any violation -> Ratio: 0.0):
  - exactly D*R + sum(course sizes) integer tokens, all finite/parseable
  - each dish's R replica ids are distinct and in range
  - per-kitchen replica count <= cap[k]  (capacity)
  - every routed kitchen for a course dish is one of that dish's own replicas

Objective (minimize): F = sum over courses of (max over kitchens of #dishes that
course routes to that kitchen)  -- the batch makespan.

Score: internal baseline B = a frequency-balanced block placement (every kitchen
hosts almost exactly the same number of replicas) + naive "always serve from
replica #0" routing (no per-course flexibility at all).
  sc = min(1000, 100*B/max(1e-9,F));  print sc/1000  (trivial ~ 0.1 by construction).
"""
import sys
import math


def compute_baseline(K, R, D, B_count, courses):
    G = K // R
    if G < 1:
        G = 1
    replica0 = [0] * D
    replica_start = [((d % G) * R) for d in range(D)]
    for d in range(D):
        replica0[d] = replica_start[d] % K  # first replica of the balanced placement
    total = 0
    for dishes in courses:
        if not dishes:
            continue
        load = {}
        for d in dishes:
            k = replica0[d]
            load[k] = load.get(k, 0) + 1
        total += max(load.values())
    return total


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0 # bad invocation")
        return 0
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path) as f:
        itoks = f.read().split()
    ii = 0

    def inext():
        nonlocal ii
        v = itoks[ii]
        ii += 1
        return v

    K = int(inext())
    R = int(inext())
    D = int(inext())
    Bc = int(inext())
    cap = [int(inext()) for _ in range(K)]
    courses = []
    for _ in range(Bc):
        s = int(inext())
        dishes = [int(inext()) for _ in range(s)]
        courses.append(dishes)

    try:
        with open(out_path) as f:
            otoks = f.read().split()
    except Exception:
        print("Ratio: 0.0 # cannot read output")
        return 0

    oi = 0
    n_out = len(otoks)

    def onext():
        nonlocal oi
        if oi >= n_out:
            raise ValueError("unexpected EOF in participant output")
        v = otoks[oi]
        oi += 1
        return v

    try:
        replica = []
        for d in range(D):
            rs = []
            for _ in range(R):
                tok = onext()
                # int() rejects "nan"/"inf"/floats/garbage; no OverflowError in py3
                iv = int(tok)
                rs.append(iv)
            if len(set(rs)) != R:
                print(f"Ratio: 0.0 # dish {d}: replica kitchens not distinct: {rs}")
                return 0
            for k in rs:
                if not (0 <= k < K):
                    print(f"Ratio: 0.0 # dish {d}: kitchen {k} out of range [0,{K})")
                    return 0
            replica.append(rs)

        load = [0] * K
        for d in range(D):
            for k in replica[d]:
                load[k] += 1
        for k in range(K):
            if load[k] > cap[k]:
                print(f"Ratio: 0.0 # capacity exceeded at kitchen {k}: {load[k]} > {cap[k]}")
                return 0

        total = 0
        for c, dishes in enumerate(courses):
            if not dishes:
                continue
            course_load = {}
            for d in dishes:
                tok = onext()
                iv = int(tok)
                if not (0 <= iv < K):
                    print(f"Ratio: 0.0 # course {c} dish {d}: routed kitchen {iv} out of range")
                    return 0
                if iv not in replica[d]:
                    print(f"Ratio: 0.0 # course {c} dish {d}: routed to non-replica kitchen {iv}")
                    return 0
                course_load[iv] = course_load.get(iv, 0) + 1
            total += max(course_load.values())

        if oi != n_out:
            print(f"Ratio: 0.0 # trailing garbage after expected output ({n_out - oi} extra tokens)")
            return 0
    except (ValueError, IndexError) as e:
        print(f"Ratio: 0.0 # parse/validation error: {e}")
        return 0

    F = total
    if F <= 0:
        print("Ratio: 0.0 # nonpositive objective")
        return 0

    Bbase = compute_baseline(K, R, D, Bc, courses)
    if Bbase <= 0:
        Bbase = 1

    sc = min(1000.0, 100.0 * Bbase / max(1e-9, float(F)))
    print(f"F={F} baseline={Bbase} Ratio: {sc / 1000.0:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
