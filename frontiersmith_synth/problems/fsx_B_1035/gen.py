import sys, random


def is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
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


def divisors_from_factor(f):
    divs = [1]
    for q, e in f.items():
        new = []
        qe = 1
        for _ in range(e + 1):
            for dd in divs:
                new.append(dd * qe)
            qe *= q
        divs = new
    return sorted(set(divs))


def primitive_root(p):
    n = p - 1
    f = factorize(n)
    pf = list(f.keys())
    g = 2
    while True:
        if all(pow(g, n // q, p) != 1 for q in pf):
            return g
        g += 1


def pick_prime(testId):
    base = 250 + 200 * testId
    cand = base | 1
    while True:
        if is_prime(cand):
            n = cand - 1
            f = factorize(n)
            divs = divisors_from_factor(f)
            if len(divs) >= 10 and max(f.values()) <= 6:
                return cand, divs
        cand += 2


def gen_instance(testId):
    p, divs = pick_prime(testId)
    n = p - 1
    rnd = random.Random(1000003 * testId + 7)
    g = primitive_root(p)

    cands = [d for d in divs if 10 <= d and n // d >= 6]
    if not cands:
        cands = [d for d in divs if d not in (1, n)]
    d_true = rnd.choice(cands)
    ncos = n // d_true
    r_true = pow(g, n // d_true, p)

    nclusters = min(ncos, rnd.randint(3, 5))
    chosen = rnd.sample(range(ncos), nclusters)

    T = set()
    for idx, cid in enumerate(chosen):
        leader = pow(g, cid, p)
        elems = [(leader * pow(r_true, k, p)) % p for k in range(d_true)]
        holes = set()
        if idx == 0 and d_true >= 6:
            nh = rnd.randint(1, max(1, d_true // 5))
            holes = set(rnd.sample(range(d_true), nh))
        keep = [elems[k] for k in range(d_true) if k not in holes]
        T.update(keep)

    chosen_keys = set(pow(pow(g, cid, p), d_true, p) for cid in chosen)
    pool = [x for x in range(1, p) if x not in T and pow(x, d_true, p) not in chosen_keys]
    nnoise = rnd.randint(2, 4)
    noise = rnd.sample(pool, min(nnoise, len(pool)))
    T.update(noise)

    alpha = rnd.randint(3, 8)
    return p, alpha, sorted(T)


def main():
    testId = int(sys.argv[1])
    p, alpha, T = gen_instance(testId)
    print(p, alpha)
    print(len(T))
    print(" ".join(str(x) for x in T))


if __name__ == "__main__":
    main()
