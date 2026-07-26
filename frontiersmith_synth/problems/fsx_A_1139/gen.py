import sys

# gen.py <testId> -- prints ONE Number-Theoretic-Transform (NTT) routing instance.
#
# n is a product of 2 or 3 PAIRWISE COPRIME small primes (never a prime power, never
# power-of-two) so the length is genuinely mixed-radix with coprime factors -- exactly
# the regime where Good-Thomas prime-factor re-indexing removes ALL inter-stage
# twiddles, while a textbook (any-split) Cooley-Tukey still pays them because it does
# not special-case coprimality. testId 10 uses THREE coprime factors (a compounding
# "regime change": naive multi-stage CT pays two independent twiddle layers there).
#
# Instance = (n, q, w): q is a prime with q === 1 (mod n) and w is an explicit
# primitive n-th root of unity mod q (found by deterministic search from testId only
# -- no RNG needed at all).

TEST_NS = [15, 21, 33, 35, 39, 55, 65, 77, 91, 105]


def is_prime(x):
    if x < 2:
        return False
    if x % 2 == 0:
        return x == 2
    d = 3
    while d * d <= x:
        if x % d == 0:
            return False
        d += 2
    return True


def factorize(x):
    f = {}
    d = 2
    while d * d <= x:
        while x % d == 0:
            f[d] = f.get(d, 0) + 1
            x //= d
        d += 1
    if x > 1:
        f[x] = f.get(x, 0) + 1
    return f


def find_q(n, start):
    q = start
    while True:
        if q % n == 1 and is_prime(q):
            return q
        q += 1


def find_generator(q):
    factors = factorize(q - 1)
    g = 2
    while True:
        ok = True
        for p in factors:
            if pow(g, (q - 1) // p, q) == 1:
                ok = False
                break
        if ok:
            return g
        g += 1


def main():
    tid = int(sys.argv[1])
    n = TEST_NS[(tid - 1) % len(TEST_NS)]
    start = 2_000_003 + 104_729 * tid
    q = find_q(n, start)
    g = find_generator(q)
    w = pow(g, (q - 1) // n, q)
    assert pow(w, n, q) == 1
    for p in factorize(n):
        assert pow(w, n // p, q) != 1
    sys.stdout.write("%d %d %d\n" % (n, q, w))


if __name__ == "__main__":
    main()
