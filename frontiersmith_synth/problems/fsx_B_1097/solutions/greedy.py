# TIER: greedy
"""The obvious recipe: find the ONE length that scores best on its own
(weight[l] * min(cap[l], a**l, T)) and output all reachable strings of just that
length, using the FULL alphabet. Same-length distinct strings are automatically an
antichain, so this is always feasible -- but it never spends budget on any other
length, even when the chosen length's cap is exhausted long before T or the
alphabet is. That leftover budget/alphabet is exactly what solutions/strong.py
recovers by mixing lengths."""
import sys
import itertools


def first_n_strings(a, l, n):
    out = []
    for tup in itertools.product(range(a), repeat=l):
        if len(out) >= n:
            break
        out.append("".join(str(d) for d in tup))
    return out


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    a = int(next(it))
    Lmax = int(next(it))
    T = int(next(it))
    weight = [int(next(it)) for _ in range(Lmax)]  # weight[i] = weight of length i+1
    cap = [int(next(it)) for _ in range(Lmax)]

    best_l, best_v, best_cnt = 1, -1, 0
    for l in range(1, Lmax + 1):
        cnt = min(cap[l - 1], a ** l, T)
        v = weight[l - 1] * cnt
        if v > best_v:
            best_v, best_l, best_cnt = v, l, cnt

    out = first_n_strings(a, best_l, best_cnt)
    print(len(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
