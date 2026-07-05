# TIER: strong
# Base-3 two-digit set (a.k.a. the Stanley / greedy-3AP-free sequence, OEIS A005836):
# integers in [1,n] whose base-3 representation uses only digits {0,1}. This is the
# densest of the simple digit constructions (~n^0.63) and is 3AP-free. It substantially
# beats the base-4/base-5 tiers, but is still far from the (unknown) true maximum, so it
# leaves headroom below Ratio 1.0.
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
    s = two_digit_set(n, 3)
    print(" ".join(map(str, s)))


if __name__ == "__main__":
    main()
