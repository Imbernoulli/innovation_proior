# TIER: greedy
# The "obvious" sommelier recipe: taste light-to-dark, a single monotone
# ascending sweep by true intensity, which minimizes the jump between any
# two CONSECUTIVE samples (so it looks like it's controlling contrast
# carryover perfectly). It spends its cleanser budget on a naive, blind
# schedule -- evenly spaced through the flight, ignoring which samples are
# flagships. It never considers that a long monotone run marches the
# adaptation EMA steadily away from center and into saturation, crushing
# gain (and any flagship caught inside that run) for the back half of the
# run.
import sys


def sorted_order_evenly_spaced_cleansers(N, K, V):
    order = sorted(range(1, N + 1), key=lambda i: (V[i], i))
    if K <= 0:
        return order
    out = []
    gap = max(1, N // (K + 1))
    cleansers_left = K
    for pos, idx in enumerate(order, start=1):
        out.append(idx)
        if cleansers_left > 0 and pos % gap == 0 and pos != N:
            out.append(0)
            cleansers_left -= 1
    return out


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); K = int(next(it))
    ALPHA = int(next(it)); D = int(next(it))
    CC_NUM = int(next(it)); CC_DEN = int(next(it)); CENTER = int(next(it))
    V = [0] * (N + 1)
    for i in range(1, N + 1):
        V[i] = int(next(it))
    W = [0] * (N + 1)
    for i in range(1, N + 1):
        W[i] = int(next(it))

    out = sorted_order_evenly_spaced_cleansers(N, K, V)
    print(" ".join(map(str, out)))


if __name__ == "__main__":
    main()
