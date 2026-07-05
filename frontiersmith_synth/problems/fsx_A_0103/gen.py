#!/usr/bin/env python3
"""gen.py <testId>  -- print ONE instance of the polar-research-base packing problem.

testId 1..10 is a difficulty ladder: N (number of domes) grows, and the antenna
keep-out radius r0 varies deterministically. Everything is a pure function of testId
(no randomness) so the instance is reproducible.

Instance format (stdout):
    N r0
where N domes must be packed into the unit-square ice pad [0,1]^2, each dome a circle
that must stay inside the pad, must not overlap the central antenna keep-out disk of
radius r0 centered at (0.5,0.5), and must not overlap another dome.
"""
import sys


def main():
    tid = int(sys.argv[1])
    if tid < 1:
        tid = 1
    N = 20 + 5 * tid           # 25 .. 70 domes  (large-scale family)
    r0 = 0.10 + 0.01 * (tid % 6)   # 0.10 .. 0.15 antenna keep-out radius
    print("%d %.4f" % (N, r0))


if __name__ == "__main__":
    main()
