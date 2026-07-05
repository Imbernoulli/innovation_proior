#!/usr/bin/env python3
"""gen.py <testId>  ->  one asteroid-belt survey instance on stdout.

Format C, family low-discrepancy-pointset, theme "asteroid mining".
The belt is normalized to the unit square [0,1]^2 (a flat rectified map of
the mining region).  K survey rigs are ALREADY deployed (the "anchor"
stations, clustered around a known ore vein in the lower-left).  You must
place the remaining N_free = M - K sampling probes so that the FULL set of
M stations covers the belt as uniformly as possible, i.e. minimizes the
exact 2-D star discrepancy of the union.

testId 1..10 is a difficulty ladder: more probes + more entrenched anchors.
Everything is seeded by testId only -> deterministic.
"""
import sys, random


def instance(testId):
    t = int(testId)
    nf = 32 + 16 * t              # free probes to place
    K = max(4, nf // 8)          # pre-deployed anchor rigs
    M = nf + K
    rng = random.Random(1000 + t)
    anchors = [(rng.uniform(0.02, 0.40), rng.uniform(0.02, 0.40))
               for _ in range(K)]
    return M, K, anchors


def main():
    testId = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    M, K, anchors = instance(testId)
    out = ["2 %d %d" % (M, K)]
    for (x, y) in anchors:
        out.append("%.17g %.17g" % (x, y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
