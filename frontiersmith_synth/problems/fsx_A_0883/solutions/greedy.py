# TIER: greedy
# The obvious single textbook pass: for each symbol, find its single most-common
# predecessor in one scan of T, then sort symbols by that predecessor id (so symbols
# that share a dominant predecessor end up adjacent in the alphabet order).  No search,
# no refinement, no rotation -- just one greedy statistical guess, r=0.
import sys
from collections import Counter


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    k = int(next(it))
    seq = [int(next(it)) for _ in range(n)]

    pred_count = [Counter() for _ in range(k)]
    for i in range(n):
        a = seq[i]
        p = seq[i - 1]  # i=0 wraps to seq[-1], a reasonable one-pass approximation
        pred_count[a][p] += 1

    dominant = []
    for s in range(k):
        if pred_count[s]:
            dominant.append(pred_count[s].most_common(1)[0][0])
        else:
            dominant.append(s)

    order = sorted(range(k), key=lambda s: (dominant[s], s))
    print(" ".join(map(str, order)))
    print(0)


if __name__ == "__main__":
    main()
