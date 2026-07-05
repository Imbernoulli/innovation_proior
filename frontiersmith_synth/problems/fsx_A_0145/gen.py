#!/usr/bin/env python3
"""gen.py <testId> -> prints ONE instance.

Instance format (stdin for the solver):
    line 1:  n V
where n = number of inverter time-slots (length of the emission step vector)
and V = maximum integer amplitude allowed per slot.

testId 1..10 is a difficulty ladder: n grows from tiny to the largest
'small'-scale instance. Everything is a pure deterministic function of testId.
"""
import sys

# difficulty ladder (n grows); V fixed so amplitudes have plenty of resolution
NS = [6, 8, 10, 14, 18, 24, 30, 40, 50, 64]
V = 1000


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if t < 1:
        t = 1
    if t > len(NS):
        t = len(NS)
    n = NS[t - 1]
    sys.stdout.write("%d %d\n" % (n, V))


if __name__ == "__main__":
    main()
