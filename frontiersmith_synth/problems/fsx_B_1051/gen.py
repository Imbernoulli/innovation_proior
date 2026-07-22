#!/usr/bin/env python3
"""gen.py <testId> -> prints one instance "n1 n2" to stdout.

Difficulty ladder testId 1..10, small -> large/adversarial. The (n1,n2) pairs
are chosen to sweep all three Sylow-2 regimes of G = Z_n1 x Z_n2:
  - both odd            (Sylow-2 trivial)          -> ceiling not attainable
  - exactly one even    (Sylow-2 cyclic nontrivial) -> ceiling attainable
  - both even            (Sylow-2 non-cyclic)        -> ceiling not attainable
Cases 5..10 are the planted traps: a single-objective greedy pass gets a
respectable partial-sum count but its banner-gaps collapse toward O(sqrt(n))
distinct values, landing far below a regime-aware joint construction.
Deterministic: purely a function of testId, no external randomness.
"""
import sys

CASES = {
    1: (3, 2),     # n=6,   one even  -> good regime, tiny sanity
    2: (5, 4),     # n=20,  one even  -> good regime, small
    3: (5, 3),     # n=15,  both odd  -> defect regime, small
    4: (6, 4),     # n=24,  both even -> defect regime, small-medium
    5: (9, 5),     # n=45,  both odd  -> defect regime, TRAP
    6: (16, 9),    # n=144, one even  -> good regime, TRAP
    7: (8, 12),    # n=96,  both even -> defect regime, TRAP
    8: (21, 11),   # n=231, both odd  -> defect regime, TRAP (large)
    9: (32, 15),   # n=480, one even  -> good regime, TRAP (large)
    10: (25, 27),  # n=675, both odd  -> defect regime, TRAP (largest)
}


def main():
    t = int(sys.argv[1])
    n1, n2 = CASES[t]
    print(n1, n2)


if __name__ == "__main__":
    main()
