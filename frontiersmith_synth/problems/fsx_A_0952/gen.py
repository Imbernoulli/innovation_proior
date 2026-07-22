#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE training ledger to stdout.

Skin: a commercial kitchen's undocumented batch-prep routine recursively splits
a catering order of n portions into a "stockpot batch" of floor(n/A) portions
and a "garnish batch" of ceil(n/B) portions (each handled by the SAME routine,
recursively), plus a fixed correction number of extra knife/plate operations
that depends only on n mod M (a spice-rotation rule baked into the routine),
plus a constant number of operations per portion. The exact prep-op count for
every order size the deli has ever run is logged in the paper ledger.

Hidden law (NEVER printed): divisors A, B; modulus M; correction table
g[0..M-1]; per-portion constant C; base counts T(0), T(1). For n>=2:
    T(n) = T(floor(n/A)) + T(ceil(n/B)) + g[n % M] + C*n
Base case: T(0), T(1) given directly (n=1 is a fixed point of ceil(n/B) for
ANY B, so it must be a base case, never a recursive one).

STDOUT prints ONLY: a header "<n_train> <test_id>" then n_train+1 rows
"<n> <T(n)>" for n = 0..n_train. The divisors, modulus, correction table,
per-portion constant, and RNG seed are NOT printed anywhere.
"""
import sys, random, hashlib

PAIRS = [(2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (2, 6), (3, 6), (4, 5)]


def _law_seed(t):
    """Non-linear per-test-id seed (SHA-256 of a salted tag), so the hidden law
    cannot be reconstructed by guessing a simple arithmetic seed formula even
    if the exact multiplier/offset convention used elsewhere were known."""
    tag = "fsx_A_0952-batchprep-law-v1:%d" % t
    return int.from_bytes(hashlib.sha256(tag.encode()).digest()[:8], "big")


def truth_params(t):
    """Hidden batch-prep law for this test id (duplicated in gen.py AND verify.py,
    never printed, never importable)."""
    rng = random.Random(_law_seed(t))
    a, b = rng.choice(PAIRS)
    m = rng.choice([4, 5, 6, 7, 8])
    g = [rng.randint(-5, 5) for _ in range(m)]
    c = rng.randint(1, 3)
    base0 = rng.randint(1, 9)
    base1 = rng.randint(1, 9)
    return a, b, m, g, c, base0, base1


def build_table(n_train, a, b, m, g, c, base0, base1):
    """Bottom-up DP: T(0..n_train)."""
    T = [0] * (n_train + 1)
    T[0] = base0
    if n_train >= 1:
        T[1] = base1
    for n in range(2, n_train + 1):
        lo = n // a
        hi = -(-n // b)  # ceil(n/b)
        T[n] = T[lo] + T[hi] + g[n % m] + c * n
    return T


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if t < 1:
        t = 1

    a, b, m, g, c, base0, base1 = truth_params(t)

    # difficulty ladder: fewer training rows and a wider extrapolation gap
    # for larger testId (kept modest: only affects size, not the mechanism).
    n_train = 900 - 55 * (t - 1)
    if n_train < 340:
        n_train = 340

    T = build_table(n_train, a, b, m, g, c, base0, base1)

    out = ["%d %d" % (n_train, t)]
    for n in range(n_train + 1):
        out.append("%d %d" % (n, T[n]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
