#!/usr/bin/env python3
"""gen.py <testId> -- prints one weighing-design instance "n k" to stdout.
testId = 1..10 is a difficulty ladder (small -> large/adversarial), fully fixed
(no randomness needed: every case is deterministic by construction)."""
import sys

# (n, k) ladder. Several cases (5, 8, 9) hide a MULTI-block structure inside a
# single n (the obvious single-pass greedy sees only one n x n grid and has no
# reason to suspect a repeated sub-pattern). Two cases (7, 10) are deliberately
# "off-size" so no clean block tiling exists -- these keep the problem genuinely
# open-ended (no reachable perfect optimum) instead of degenerating into a
# lookup table.
CASES = {
    1:  (4,   3),
    2:  (8,   7),
    3:  (12,  11),
    4:  (20,  19),
    5:  (24,  11),
    6:  (24,  23),
    7:  (53,  23),
    8:  (60,  19),
    9:  (64,  31),
    10: (108, 43),
}


def main():
    tid = int(sys.argv[1])
    if tid not in CASES:
        tid = ((tid - 1) % 10) + 1
    n, k = CASES[tid]
    print(n, k)


if __name__ == "__main__":
    main()
