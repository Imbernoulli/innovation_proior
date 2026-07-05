#!/usr/bin/env python3
# gen.py <testId>  ->  prints ONE instance (n r) to stdout.
# testId 1..10 is a difficulty ladder: growing dimension n and covering radius r.
# The instance is fully determined by testId (no randomness) -> deterministic.
import sys

# (n, r) ladder.  Fault-tolerant-memory theme: a memory word is n bits; a covering
# code of radius r is a set of "anchor" words so that every possible stored word is
# within r bit-flips of some anchor (r-fault correctable to an anchor).
LADDER = {
    1: (8, 2),
    2: (9, 2),
    3: (10, 2),
    4: (11, 2),
    5: (12, 2),
    6: (9, 3),
    7: (10, 3),
    8: (11, 3),
    9: (12, 3),
    10: (13, 2),
}


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    t = int(sys.argv[1])
    if t not in LADDER:
        # clamp into range so the harness never gets an empty instance
        t = ((t - 1) % len(LADDER)) + 1
    n, r = LADDER[t]
    print("%d %d" % (n, r))


if __name__ == "__main__":
    main()
