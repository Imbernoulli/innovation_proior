import sys, random

# gen.py <testId>  -- prints ONE reversible-synthesis instance to stdout.
#
# The target is a modular permutation polynomial over Z_{2^n}:
#     pi(x) = (c0 + c1*x + c2*x^2 + c3*x^3) mod 2^n
# By Rivest's criterion this permutes Z_{2^n} iff c1 is odd and c2, c3 are even;
# c0 in [1, 2^n) guarantees pi is NOT the identity (pi(0) = c0 != 0). These maps
# are the building blocks of modular-arithmetic (quantum) circuits and are
# genuinely nonlinear over GF(2), so no closed-form minimal MCT circuit is known.
#
# testId 1..10 is the difficulty ladder: register width n grows 4 -> 8.
NLAD = {1: 4, 2: 5, 3: 5, 4: 6, 5: 6, 6: 7, 7: 7, 8: 8, 9: 8, 10: 8}
SEED = 424242


def main():
    tid = int(sys.argv[1])
    n = NLAD[tid]
    N = 1 << n
    rng = random.Random(SEED + 1000 * tid)
    c0 = rng.randrange(1, N)        # nonzero constant -> not identity
    c1 = rng.randrange(1, N, 2)     # odd
    c2 = rng.randrange(0, N, 2)     # even
    c3 = rng.randrange(0, N, 2)     # even
    perm = [(c0 + c1 * x + c2 * x * x + c3 * x * x * x) % N for x in range(N)]
    # sanity: it is a permutation
    assert sorted(perm) == list(range(N))
    out = [str(n), " ".join(str(v) for v in perm)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
