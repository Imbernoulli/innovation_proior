#!/usr/bin/env python3
"""gen.py <testId> -> one instance on stdout.

Instance: "M D k W" on a single line.

    M  - the full channel modulus (Z_M).
    D  - a small "coarse band" divisor of M, with gcd(D, M/D) = 1.
    k  - number of channels to select (a k-subset of Z_M).
    W  - positive integer weight on the coarse-band term (see statement.md).

testId 1..10 indexes a fixed difficulty ladder (D, M0, k) with M = D * M0. Everything is
determined purely by testId (no external randomness), so generation is bit-for-bit
reproducible. Each M0 is chosen coprime to D and comfortably larger than 2*k^2, so a true
Sidon set of size k exists inside the M0-quotient (needed by the intended strong solution);
D is deliberately kept SMALLER than k, so the coarse band always forces repeated residues
(pigeonhole) -- this is the trap: a flat search that only fights collisions in the full
group Z_M has no reason to also keep that forced repetition well spread across the D bands,
and pays for it heavily once the coarse term is weighted in.
"""
import sys

# (D, M0, k) ladder, small -> large/adversarial. gcd(D, M0) = 1 and M0 > 2*k*k always.
LADDER = [
    (4, 53, 5),
    (4, 73, 6),
    (6, 101, 7),
    (6, 131, 8),
    (8, 163, 9),
    (8, 211, 10),
    (8, 251, 11),
    (8, 293, 12),
    (8, 347, 13),
    (8, 397, 14),
]


def energy_of_counts(counts, d):
    """Additive energy of a residue-count vector over Z_d (exact int)."""
    r = [0] * d
    for i in range(d):
        ci = counts[i]
        if ci == 0:
            continue
        for j in range(d):
            cj = counts[j]
            if cj:
                r[(i + j) % d] += ci * cj
    return sum(x * x for x in r)


def balanced_counts(d, k):
    """The (provably energy-minimal) as-equal-as-possible residue counts summing to k."""
    base, rem = divmod(k, d)
    return [base + 1 if i < rem else base for i in range(d)]


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    t = int(sys.argv[1])
    t = max(1, min(len(LADDER), t))
    D, M0, k = LADDER[t - 1]
    M = D * M0

    B0 = 2 * k * k - k
    Ed_floor = energy_of_counts(balanced_counts(D, k), D)
    W = max(1, round(4 * B0 / Ed_floor))

    sys.stdout.write("%d %d %d %d\n" % (M, D, k, W))


if __name__ == "__main__":
    main()
