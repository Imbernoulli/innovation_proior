# TIER: strong
# Insight: adaptation and contrast pull in OPPOSITE directions, but only
# when there is enough range to drive the adaptation EMA far from center in
# the first place. So first measure the instance's own risk: the spread of
# true intensities present. If that spread cannot push the EMA past the
# gain-saturation elbow no matter how it is swept, the monotone sommelier
# sweep is *already* the right call (its small consecutive jumps minimize
# contrast damage for free, and there is no saturation risk to hedge
# against) -- so fall back to exactly that recipe.
#
# Only when the spread is big enough to threaten saturation does the real
# reformulation kick in: split the sorted samples into a low half L and a
# high half H, and pair them by matching PERCENTILE rank -- L[i] with
# H[i], not global extremes. Interleaving L[0],H[0],L[1],H[1],... makes the
# EMA swing back toward center every two steps (gain never saturates)
# while each individual jump is only a percentile-matched gap, not the
# full range (so contrast damage stays bounded). The cleanser budget is
# then spent immediately before the highest-weight flagship samples,
# resetting adaptation to neutral right when it matters most.
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


def band_limited_zigzag_with_flagship_cleansers(N, K, V, W):
    S = sorted(range(1, N + 1), key=lambda i: (V[i], i))
    half = N // 2
    L = S[:half]
    H = S[half:]
    m = min(len(L), len(H))
    zigzag = []
    for i in range(m):
        zigzag.append(L[i])
        zigzag.append(H[i])
    if len(H) > m:
        zigzag.extend(H[m:])
    if len(L) > m:
        zigzag.extend(L[m:])

    protect_n = min(K, N)
    protect = set(sorted(range(1, N + 1), key=lambda i: (-W[i], i))[:protect_n])

    out = []
    budget = K
    for idx in zigzag:
        if idx in protect and budget > 0:
            out.append(0)
            budget -= 1
        out.append(idx)
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

    Vs = [V[i] for i in range(1, N + 1)]
    spread = max(Vs) - min(Vs)
    risk_threshold = 2.5 * (D / 1000.0)   # elbow-relative: below this, no
                                            # sweep can push the EMA past
                                            # the saturation elbow D

    if spread <= risk_threshold:
        out = sorted_order_evenly_spaced_cleansers(N, K, V)
    else:
        out = band_limited_zigzag_with_flagship_cleansers(N, K, V, W)

    print(" ".join(map(str, out)))


if __name__ == "__main__":
    main()
