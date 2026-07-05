# TIER: greedy
# Greedy column-by-column insertion: for each new column choose the unused row
# that adds the fewest new displacement-vector coincidences with the columns
# placed so far (ties -> smallest row). Deterministic; a solid single-pass
# construction that is far better than the identity but leaves clear headroom.
import sys
from collections import Counter


def greedy_insert(n):
    used = [False] * n
    p = []
    cnt = Counter()  # multiplicities of displacement vectors placed so far
    for col in range(n):
        best_v = None
        best_add = None
        best_new = None
        for v in range(n):
            if used[v]:
                continue
            add = 0
            newv = Counter()
            for k in range(col):
                vec = (col - k, v - p[k])
                if cnt[vec] + newv[vec] >= 1:
                    add += 1
                newv[vec] += 1
            if best_add is None or add < best_add:
                best_add = add
                best_v = v
                best_new = newv
        p.append(best_v)
        used[best_v] = True
        cnt.update(best_new)
    return p


def main():
    n = int(sys.stdin.read().split()[0])
    p = greedy_insert(n)
    print(" ".join(str(x) for x in p))


if __name__ == "__main__":
    main()
