# TIER: trivial
# Reproduces the checker's reference construction: integers in [1,n] whose base-5
# representation uses only digits {0,1}. This is 3AP-free (no-carry argument) but sparse,
# so it scores ~0.1 (it equals the baseline the checker normalizes against).
import sys


def two_digit_set(n, b):
    powers = []
    p = 1
    while p <= n:
        powers.append(p)
        p *= b
    L = len(powers)
    res = []
    for mask in range(1, 1 << L):
        s = 0
        m = mask
        i = 0
        while m:
            if m & 1:
                s += powers[i]
            m >>= 1
            i += 1
        if 1 <= s <= n:
            res.append(s)
    return sorted(set(res))


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    s = two_digit_set(n, 5)
    print(" ".join(map(str, s)))


if __name__ == "__main__":
    main()
