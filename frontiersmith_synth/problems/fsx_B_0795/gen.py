import sys, random


def is_probable_prime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def factorize(n):
    f = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            f[d] = f.get(d, 0) + 1
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        f[n] = f.get(n, 0) + 1
    return f


def find_primitive_root(p, prime_factors_of_pm1):
    n = p - 1
    for g in range(2, p):
        ok = True
        for q in prime_factors_of_pm1:
            if pow(g, n // q, p) == 1:
                ok = False
                break
        if ok:
            return g
    raise RuntimeError("no primitive root found")


def build_params(testId, rng):
    # o: hidden-subgroup order, kept EVEN so o*Q + 1 can be odd (Q is an odd prime).
    o = 12 + 2 * testId  # 14 .. 32
    Q = 150 + 60 * testId
    while True:
        Q += 1
        if not is_probable_prime(Q):
            continue
        if Q <= o:
            continue
        p = o * Q + 1
        if is_probable_prime(p):
            return o, Q, p


def main():
    testId = int(sys.argv[1])
    rng = random.Random(202400 + 97 * testId)

    o, Q, p = build_params(testId, rng)
    factors_pm1 = list(factorize(p - 1).keys())
    g = find_primitive_root(p, factors_pm1)
    h = pow(g, Q, p)  # order exactly o
    Hset = set()
    val = 1
    for _ in range(o):
        Hset.add(val)
        val = (val * h) % p

    LAMBDA = 8
    T = 150 + 40 * testId
    d = max(3, T // 45)  # decoys ("difficult client" trap cases)

    c = rng.randrange(2, p - 1)
    cinv = pow(c, p - 2, p)

    rest = []  # (value, is_decoy) for targets 2..T, order will be shuffled
    genuine_count = T - 1 - d
    for _ in range(genuine_count):
        a = rng.randrange(0, o)
        val = (c * pow(h, a, p)) % p
        rest.append((val, False))
    made = 0
    while made < d:
        cand = rng.randrange(1, p)
        r = (cand * cinv) % p
        if r not in Hset:
            rest.append((cand, True))
            made += 1
    rng.shuffle(rest)

    targets = [c] + [v for v, _ in rest]
    assert len(targets) == T

    out = []
    out.append(str(p))
    out.append(str(g))
    out.append(str(LAMBDA))
    out.append(str(T))
    out.append(" ".join(str(x) for x in targets))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
