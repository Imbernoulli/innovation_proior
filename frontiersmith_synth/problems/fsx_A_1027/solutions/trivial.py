# TIER: trivial
import sys


def divisors(n):
    ds = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            ds.append(i)
            if i != n // i:
                ds.append(n // i)
        i += 1
    return sorted(ds)


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    p = int(next(it))
    T = int(next(it))
    m = int(next(it))
    S = [int(next(it)) for _ in range(m)]

    LAMBDA = 2
    # Narrow, naive first guess: only ever test subgroup orders in 12..25.
    window = [d for d in divisors(p - 1) if 12 <= d <= 25]

    best_F = -1
    best_d, best_c = None, None
    for d in window:
        counts = {}
        for x in S:
            v = pow(x, d, p)
            counts[v] = counts.get(v, 0) + 1
        if not counts:
            continue
        c = max(counts, key=lambda k: counts[k])
        cnt = counts[c]
        F = cnt - LAMBDA * (d - cnt)
        if F > best_F:
            best_F = F
            best_d, best_c = d, c

    if best_d is None:
        # nothing usable found: emit the empty polynomial
        print(0)
        return

    # f(x) = x^d - c  ->  terms (d, 1) and (0, (-c) mod p)
    a0 = (-best_c) % p
    print(2)
    print(best_d, 1)
    print(0, a0)


if __name__ == "__main__":
    main()
