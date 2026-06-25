import sys
from functools import lru_cache

# Independent brute force for "Collapsing a strip".
#
# Tokens a[0..n-1]. Repeatedly merge two ADJACENT tokens x, y into a single
# token x+y, gaining score x*y. After n-1 merges one token remains. Maximize
# the total score. For n <= 1 no merge happens, score is 0.
#
# Brute force directly over the recursive structure of merge orders: any full
# collapse of an interval picks SOME final split point k; the very last merge
# of the interval combines the (already-collapsed) left value sum(i..k) with
# the right value sum(k+1..j), scoring sum(i..k)*sum(k+1..j), and the two sides
# are collapsed independently and optimally. We enumerate every split with no
# DP shortcut beyond memoizing on the (i,j) interval, which is the literal
# definition of the process, so this is an obviously-correct oracle.

def solve(a):
    n = len(a)
    if n <= 1:
        return 0
    pre = [0] * (n + 1)
    for i in range(n):
        pre[i + 1] = pre[i] + a[i]

    def S(i, j):
        return pre[j + 1] - pre[i]

    @lru_cache(maxsize=None)
    def best(i, j):
        if i == j:
            return 0
        res = None
        for k in range(i, j):
            cand = best(i, k) + best(k + 1, j) + S(i, k) * S(k + 1, j)
            if res is None or cand > res:
                res = cand
        return res

    return best(0, n - 1)


def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    n = int(data[0])
    a = [int(x) for x in data[1:1 + n]]
    print(solve(a))


if __name__ == "__main__":
    main()
