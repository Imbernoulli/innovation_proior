#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE instance (the pad count n) for the apiary
landing-pad spread problem. testId 1..10 is a difficulty ladder: more pads =>
more triples => a harder Heilbronn-type packing.  The field (unit triangle) and
the reference layout seed are fixed inside the checker; the instance is just n."""
import sys


def main():
    try:
        t = int(sys.argv[1])
    except (IndexError, ValueError):
        t = 1
    if t < 1:
        t = 1
    if t > 10:
        t = 10
    # ladder: testId 1..10 -> n = 8..17
    n = 7 + t
    print(n)


if __name__ == "__main__":
    main()
