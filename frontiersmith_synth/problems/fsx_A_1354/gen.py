#!/usr/bin/env python3
"""gen.py <testId> -> one instance on stdout.

Instance = two coprime positive integers "P Q" (a line from (0,0) to (P,Q) over the
integer grid). testId 1..10 is a fixed difficulty ladder, entirely determined by testId
(no randomness at all -- fully reproducible by construction).

The ladder mixes:
  - tiny/skewed pairs (continued fraction of P/Q has one or two LARGE partial quotients:
    long straight runs of one crossing type -> naive run-length coding is competitive here),
  - "medium" pairs whose continued fraction has a small leading quotient (2 or 3) so the
    first-level run structure alone stays close to the raw length even though a deeper
    hierarchical view compresses it sharply,
  - consecutive-Fibonacci pairs (testId 4, 6, 8, 10) whose continued fraction is ALL ones --
    the classical hardest case for run-length/dictionary compressors, where every run has
    length 1 or 2 and naive coding buys almost nothing, while the slope's continued-fraction
    structure still collapses the sequence logarithmically.
"""
import sys

# (P, Q) ladder -- both coprime, increasing scale; rows 4,6,8,10 are consecutive Fibonacci
# pairs (guaranteed all-ones continued fraction => the compression trap).
LADDER = [
    (5, 2),      # 1: tiny sanity
    (9, 2),      # 2: easy, skewed (one big run)
    (100, 7),    # 3: easy, skewed (one big leading quotient)
    (55, 34),    # 4: TRAP -- consecutive Fibonacci
    (61, 17),    # 5: medium, small leading quotient
    (89, 55),    # 6: TRAP -- consecutive Fibonacci
    (137, 64),   # 7: medium, small leading quotient
    (233, 144),  # 8: TRAP -- consecutive Fibonacci
    (521, 233),  # 9: medium, small leading quotient
    (987, 610),  # 10: TRAP -- consecutive Fibonacci, largest
]


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    t = int(sys.argv[1])
    if t < 1:
        t = 1
    if t > len(LADDER):
        t = len(LADDER)
    P, Q = LADDER[t - 1]
    sys.stdout.write("%d %d\n" % (P, Q))


if __name__ == "__main__":
    main()
