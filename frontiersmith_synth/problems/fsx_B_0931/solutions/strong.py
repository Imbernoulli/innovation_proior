# TIER: strong
# Genuine AMR-style insight: rather than chasing the single largest raw
# indicator values (greedy), treat the probe samples as a proxy field and
# find the EXACT partition of the N0 base blocks into B groups that
# EQUIDISTRIBUTES the estimated indicator -- i.e. minimizes the sum, over
# groups, of the within-group variance of the probe samples. This is
# solved by an O(N0^2 * B) DP that is the closed-form fixed point of the
# "coarsen the least-informative region, recycle its cell to refine the
# most error-heavy region" loop: it simultaneously (a) ESTIMATES a local
# error indicator from the limited probe data, (b) EQUIDISTRIBUTES the
# fixed budget against that indicator instead of the raw domain length,
# and (c) implicitly COARSENS flat/smooth stretches (merges many probe
# blocks into one wide, cheap group) to RECYCLE cells toward the fronts.
import sys, json


def main():
    inst = json.load(sys.stdin)
    N0 = inst["N0"]
    B = inst["B"]
    BW = inst["BW"]
    P = inst["probe_f"]

    pre = [0.0] * (N0 + 2)
    pre2 = [0.0] * (N0 + 2)
    for i in range(N0 + 1):
        pre[i + 1] = pre[i] + P[i]
        pre2[i + 1] = pre2[i] + P[i] * P[i]

    def cost(lo, hi):  # group spans probe indices lo..hi inclusive
        n = hi - lo + 1
        s = pre[hi + 1] - pre[lo]
        s2 = pre2[hi + 1] - pre2[lo]
        return s2 - (s * s) / n

    NEG = float("inf")
    dp = [[NEG] * (N0 + 1) for _ in range(B + 1)]
    choice = [[-1] * (N0 + 1) for _ in range(B + 1)]
    dp[0][0] = 0.0
    for k in range(1, B + 1):
        for hi in range(k, N0 + 1):
            best_v, best_lo = NEG, -1
            for lo in range(k - 1, hi):
                if dp[k - 1][lo] == NEG:
                    continue
                v = dp[k - 1][lo] + cost(lo, hi)
                if v < best_v:
                    best_v, best_lo = v, lo
            dp[k][hi] = best_v
            choice[k][hi] = best_lo

    bnds = [N0]
    k, hi = B, N0
    while k > 0:
        lo = choice[k][hi]
        bnds.append(lo)
        hi = lo
        k -= 1
    bnds.reverse()
    idxs = bnds[1:-1]
    cuts = [j * BW for j in idxs]
    print(json.dumps({"cuts": cuts}))


main()
