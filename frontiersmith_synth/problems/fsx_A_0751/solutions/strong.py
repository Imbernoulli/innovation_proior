# TIER: strong
# Lattice detection + coprimality certificate.
#
# Insight: every instance's tracking codes secretly satisfy x_i = x_j (mod D)
# for a fixed BATCH STRIDE D -- D is exactly the GCD of all pairwise
# differences, and it is recoverable from the PUBLIC codes alone (no oracle
# needed). If the hash modulus M shares a prime factor with D, the periodic
# structure survives the first "mod M" stage and the trucks collapse. So:
#   1. Detect D = gcd_i(code_i - code_0).
#   2. Factor D by trial division (D is always smooth/modest here) to get its
#      prime-factor certificate.
#   3. Search upward from a large fixed base for a modulus M that shares NO
#      prime factor with D -- gcd(M, D) == 1 is a certificate that the "mod M"
#      stage cannot fold the lattice down.
#   4. Pick a multiplier coprime to M (so the "mod M" stage is a bijection on
#      its own residues) and a fixed shift.
# A static, non-adaptive hash (any fixed M chosen without looking at the data)
# cannot offer this guarantee -- it only works by luck when D happens to avoid
# M's prime factors.
import sys, json, math


def factorize(n):
    fs = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            fs.add(d)
            n //= d
        d += 1
    if n > 1:
        fs.add(n)
    return fs


def main():
    inst = json.load(sys.stdin)
    codes = inst["codes"]
    base = codes[0]

    # 1. detect the planted batch stride via pairwise-difference GCD
    D = 0
    for x in codes[1:]:
        D = math.gcd(D, abs(x - base))
        if D == 1:
            break
    if D == 0:
        D = 1

    # 2. factor it (certificate of which primes must be avoided)
    primes = factorize(D) if D > 1 else set()

    # 3. find a modulus coprime to every prime factor of D
    M = 4_000_003
    if M % 2 == 0:
        M += 1
    while any(M % p == 0 for p in primes):
        M += 2

    # 4. multiplier coprime to M, fixed shift
    a = 2654435761 % M
    if a == 0:
        a = 1
    while math.gcd(a, M) != 1 and a < M:
        a += 1
    if a >= M or a < 1:
        a = 1
    c = 999331 % M

    print(json.dumps({"a": a, "c": c, "M": M}))


main()
