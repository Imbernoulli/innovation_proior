#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE cache/tiling instance to stdout.

Instance:
    N
    C L A
    PAD_MAX

N            : matrix dimension (C = A x B, both N x N, row-major)
C            : cache capacity, in WORDS
L            : cache line size, in WORDS
A            : associativity (ways per set); S = C // (L*A) sets
PAD_MAX      : max allowed per-array row padding, in WORDS (0 <= pad <= PAD_MAX)

Deterministic: purely a function of testId (a fixed lookup table -- no RNG needed,
which keeps every rerun byte-identical).

Cases 1-3  : "warm-up" cases -- cache is fully-associative (S == 1, one giant set),
             so no cross-row conflict is even possible; only capacity/reuse and
             loop order matter.
Cases 4-10 : low-associativity cases where the cache/matrix geometry causes real
             cross-row conflict aliasing at zero padding -- a capacity-only tile
             size (that ignores associativity) thrashes here; padding is required
             to spread rows across sets and avoid it.
"""
import sys

# (N, C_words, L_words, A_ways, PAD_MAX_words)
TABLE = {
    1: (24, 192, 4, 48,  4),     # warm-up, fully-associative (S=1)
    2: (32, 300, 4, 75,  4),     # warm-up, fully-associative (S=1)
    3: (40, 504, 4, 126, 4),     # warm-up, fully-associative (S=1)
    4: (16, 16,  4, 1,  16),     # low-assoc aliasing, direct-mapped
    5: (20, 240, 4, 6,  20),     # low-assoc aliasing, 6-way
    6: (24, 48,  4, 2,  24),     # low-assoc aliasing, 2-way
    7: (28, 56,  4, 2,  28),     # low-assoc aliasing, 2-way
    8: (32, 64,  4, 2,  32),     # low-assoc aliasing, 2-way
    9: (36, 192, 4, 4,  36),     # low-assoc aliasing, 4-way
    10:(40, 80,  4, 2,  40),     # low-assoc aliasing, 2-way
}


def main():
    tid = int(sys.argv[1])
    if tid not in TABLE:
        tid = ((tid - 1) % 10) + 1
    N, C, L, A, PAD_MAX = TABLE[tid]
    assert C % (L * A) == 0, "cache geometry must divide evenly"
    print(N)
    print(C, L, A)
    print(PAD_MAX)


if __name__ == "__main__":
    main()
