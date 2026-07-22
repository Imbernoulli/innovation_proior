# TIER: greedy
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
    n = p - 1
    # Textbook / obvious move: try EVERY candidate subgroup order (an
    # exhaustive scan, not just a lucky guess) but still fit a SINGLE binomial.
    cand = [d for d in divisors(n) if 2 <= d <= n // 2]

    best_F = -1
    best_d, best_c = None, None
    for d in cand:
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

    if best_d is None or best_F <= 0:
        print(0)
        return

    a0 = (-best_c) % p
    print(2)
    print(best_d, 1)
    print(0, a0)


if __name__ == "__main__":
    main()
