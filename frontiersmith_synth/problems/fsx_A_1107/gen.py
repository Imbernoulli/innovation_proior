import sys, random

# gen.py <testId> -- prints ONE grid-evaluation instance to stdout.
#
# The hidden target function is
#     f(x) = A*(sqrt(x^2+B^2) - x) + C*(B/sqrt(x^2+B^2)) + E*log(1+exp(D*x))
# a fixed three-term composite (hypot-residual + companion + softplus) whose
# coefficients A,B,C,D,E vary per test.  testId 1..3 are TAME (moderate |x|,
# moderate D*x) -- the textbook direct formula is fully accurate there.
# testId 4..10 are TRAP tests: the grid spans the full 1e-30..1e30 dynamic
# range (both signs) via fixed extreme anchors plus randomized log-uniform
# points, which is where the direct/naive formula suffers catastrophic
# cancellation (hypot term, large positive x) and exp overflow (softplus
# term, large |D*x|).
#
# Output:
#   A B C D E
#   M
#   x_1
#   ...
#   x_M
# (all floats printed via repr() so they round-trip exactly through float()).

TAME = {1, 2, 3}


def rand_point(rng, exp_lo, exp_hi):
    exp = rng.uniform(exp_lo, exp_hi)
    mant = rng.uniform(1.0, 9.999)
    sign = rng.choice([1, -1])
    return sign * mant * (10.0 ** exp)


def main():
    tid = int(sys.argv[1])
    rng = random.Random(10000 + tid * 7919)

    A = round(rng.uniform(0.5, 3.0), 4)
    B = round(rng.uniform(0.6, 3.0), 4)
    C = round(rng.uniform(0.5, 3.0), 4)
    E = round(rng.uniform(0.5, 3.0), 4)

    if tid in TAME:
        D = round(rng.uniform(0.3, 0.9), 4)
        M = 24 + 4 * tid
        points = [rand_point(rng, -3.0, 1.15) for _ in range(M)]
    else:
        D = round(rng.uniform(0.4, 2.5), 4) * rng.choice([1, -1])
        anchor_exps = [-30, -25, -20, -15, -10, -5, -2, 0, 2, 5, 10, 15, 20, 25, 30]
        anchors = []
        for e in anchor_exps:
            v = 10.0 ** e
            anchors.append(v)
            anchors.append(-v)
        extra_n = 14 + 4 * (tid - 4)
        extra = [rand_point(rng, -30.0, 30.0) for _ in range(extra_n)]
        points = anchors + extra
        M = len(points)

    out = ["%r %r %r %r %r" % (A, B, C, D, E), str(M)]
    for p in points:
        out.append(repr(p))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
