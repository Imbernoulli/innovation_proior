#!/usr/bin/env python3
"""gen.py <testId> -> prints ONE instance to stdout.

Instance format (stdin the solver reads):
    line 1:  n k
      n = order of the Latin squares
      k = number of Latin squares to construct

The solver must output k Latin squares of order n (see statement.md), maximizing the
total number of DISTINCT superimposed ordered symbol-pairs summed over all C(k,2)
unordered pairs of squares (i.e. how close the set is to being mutually orthogonal).

testId 1..10 is a difficulty ladder over orders where NO perfect (mutually orthogonal)
construction is reachable by the trivial cyclic method (all coprime residues are even
=> no cyclic pair is orthogonal), and several are genuinely open (e.g. 3 MOLS of order 10).
"""
import sys

# (n, k) ladder: small -> large / harder.  All orders n satisfy: every residue coprime
# to n is even (n in {6,10,12,14,18,22}), so the cyclic baseline is never orthogonal.
INSTANCES = {
    1:  (6, 2),
    2:  (10, 2),
    3:  (6, 3),
    4:  (10, 3),
    5:  (12, 2),
    6:  (12, 3),
    7:  (14, 2),
    8:  (14, 3),
    9:  (18, 3),
    10: (22, 3),
}


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(2)
    t = int(sys.argv[1])
    if t not in INSTANCES:
        # clamp into range so the ladder is robust
        t = ((t - 1) % len(INSTANCES)) + 1
    n, k = INSTANCES[t]
    sys.stdout.write("%d %d\n" % (n, k))


if __name__ == "__main__":
    main()
