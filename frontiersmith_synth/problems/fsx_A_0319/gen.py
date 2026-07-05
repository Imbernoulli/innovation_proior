#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE instance of the "quantum lab wiring" low-discrepancy
probe-placement problem.

Theme: a cryogenic quantum chip is characterised by dropping M measurement probes
onto the normalised 2-D wafer square [0,1]^2.  A biased probe layout misses whole
sub-regions of the chip; the calibration bias is exactly the *star discrepancy* of
the probe set.  You choose where the M probes go so that EVERY axis-aligned corner
box [0,a) x [0,b) holds a fraction of probes close to its area a*b.

The instance is just the layout size.  testId 1..10 is a difficulty ladder
(few probes -> many probes).  Everything is seeded by testId only.

Output format (stdin the solver reads):
    d M
where d = 2 (dimension) and M = number of probes to emit.
"""
import sys

# probe-count ladder (small/fast -> large).  d is fixed to 2 (exact star
# discrepancy is only tractable for small fixed dimension).
M_LADDER = [16, 24, 32, 40, 48, 56, 64, 72, 84, 96]


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    t = max(1, min(len(M_LADDER), t))
    m = M_LADDER[t - 1]
    sys.stdout.write("2 %d\n" % m)


if __name__ == "__main__":
    main()
