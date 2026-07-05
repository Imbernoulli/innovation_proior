# TIER: strong
# More-sums-than-differences construction. Start from the classic MSTD seed
# B0 = {0,2,3,4,7,11,12,14} with |B0+B0|=26 > |B0-B0|=25 (ratio 26/25 = 1.04).
# Direct sum with a well-separated factor C multiplies BOTH set sizes, so the ratio
# multiplies: A = B0 (+) d*C has ratio 1.04 * ratio(C), with |A| = 8*|C|.
#   - n a multiple of 64: take C itself an MSTD block -> ratio 1.04^2 = 1.0816.
#   - n a multiple of 8 : take C an arithmetic run (ratio 1) -> ratio 1.04.
# Either way |A| = n exactly and the ratio strictly exceeds the greedy run (=1).
import sys

B0 = [0, 2, 3, 4, 7, 11, 12, 14]


def span(A):
    return max(A) - min(A)


def dsum(A, B):
    # A (+) d*B with d = 2*span(A)+1, large enough that the sumsets AND difference
    # sets of the two factors never collide -> both cardinalities are multiplicative.
    d = 2 * span(A) + 1
    return sorted({a + d * b for a in A for b in B})


def build(n):
    if n % 64 == 0:
        C = dsum(B0, list(range(n // 64)))   # size n/8, ratio 1.04
    else:
        C = list(range(n // 8))              # size n/8, ratio 1.0
    return dsum(B0, C)                        # size 8*|C| = n


def main():
    toks = sys.stdin.read().split()
    n, M = int(toks[0]), int(toks[1])
    A = build(n)
    # safety: stay inside [0, M] (holds for the given M = 4 n^2)
    assert len(set(A)) == n and max(A) <= M and min(A) >= 0
    print(" ".join(map(str, A)))


if __name__ == "__main__":
    main()
