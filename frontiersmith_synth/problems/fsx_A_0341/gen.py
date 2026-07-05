#!/usr/bin/env python3
"""
gen.py <testId> -> prints ONE instance of the DEEP-SEA CABLE INTERFERENCE-FREE
BACKBONE problem to stdout.

Instance semantics (theme skin):
  A deep-sea cable operator must choose a set of candidate cable *routes*.  Each
  candidate route is encoded as a length-n string of trits (a vector in F_3^n),
  the "signature" of which relay chain it uses.  Route v carries an integer
  throughput value w(v) >= 1.  Three distinct routes a,b,c mutually interfere
  (a destructive resonance) exactly when a+b+c == 0 (mod 3) coordinate-wise --
  i.e. the three signatures are collinear in the affine space AG(n,3).  The
  operator may only deploy a set of routes that contains NO such interfering
  triple (an "interference-free backbone", a *cap set*).  Deploy a backbone of
  maximum total throughput.

STDOUT format:
  line 1 : n
  line 2 : 3^n integers, the throughput w of every route, listed in canonical
           base-3 order (route index i has trits = base-3 digits of i, most
           significant coordinate first).

Difficulty ladder (testId 1..10): n grows 4 -> 7, seed varies per test.
Randomness is seeded ONLY by testId, so generation is fully deterministic.
"""
import sys, random

# testId -> dimension n (small = fast reward, large = evaluation)
LADDER = [4, 4, 5, 5, 6, 6, 6, 7, 7, 7]


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py <testId>\n")
        sys.exit(1)
    t = int(sys.argv[1])
    if t < 1:
        t = 1
    if t > len(LADDER):
        # extend deterministically beyond the table if ever asked
        n = 7
    else:
        n = LADDER[t - 1]

    N = 3 ** n
    rnd = random.Random(t * 1000003 + n * 7 + 11)
    # skewed integer weights in [1, 1000]: most routes cheap, a few very valuable
    w = [1 + int((rnd.random() ** 2) * 999) for _ in range(N)]

    out = sys.stdout
    out.write("%d\n" % n)
    out.write(" ".join(map(str, w)))
    out.write("\n")


if __name__ == "__main__":
    main()
