#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE checksum-placement-design instance to stdout.

Difficulty ladder testId 1..10: message length M grows, and the probability
mass on burst-mode corruption (PBURST) grows, so later cases are dominated
by the bursty channel behaviour the real system actually exhibits.

Output (stdout):
    M K SEED PBURST
    NBL
    L_1 W_1
    ...
    L_NBL W_NBL

All values are plain, deterministic functions of testId -- no randomness is
used at generation time (the held-out error patterns used for scoring are
generated later, inside the checker, seeded from SEED read out of this
file).
"""
import sys

# Fixed overhead budget: K checksum (parity) groups for every instance.
K = 6

# Burst-length menu (public). Weighted mixture of lengths that do NOT divide
# K evenly (routed well by interleaving) and a few lengths that ARE multiples
# of K with an even quotient (hard for every scheme, incl. interleaving --
# keeps headroom open). All lengths are even, so a lone parity bit over the
# whole message (or a burst fully inside one wide block) never sees them.
LENGTH_TABLE = [
    (8, 3), (10, 3), (14, 3), (16, 3),
    (20, 2), (22, 2), (26, 2), (28, 2),
    (12, 2), (24, 2), (36, 1),
]

# Ladder: message length M (multiple of K) and PBURST (share of held-out
# corruption events that are bursts rather than scattered independent bits).
LADDER = [
    #  M     PBURST
    (120, 0.15),
    (150, 0.20),
    (180, 0.25),
    (222, 0.40),
    (258, 0.50),
    (300, 0.60),
    (342, 0.65),
    (378, 0.70),
    (420, 0.72),
    (456, 0.75),
]


def main():
    if len(sys.argv) != 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    test_id = int(sys.argv[1])
    if not (1 <= test_id <= len(LADDER)):
        print("testId out of range", file=sys.stderr)
        sys.exit(1)

    M, pburst = LADDER[test_id - 1]
    seed = test_id  # echoed verbatim; the checker derives its own hidden RNG from it

    out = []
    out.append(f"{M} {K} {seed} {pburst:.4f}")
    out.append(str(len(LENGTH_TABLE)))
    for L, W in LENGTH_TABLE:
        out.append(f"{L} {W}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
