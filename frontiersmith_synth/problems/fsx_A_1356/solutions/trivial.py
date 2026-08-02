# TIER: trivial
"""
Ignore the historical log entirely. Assume the opponent plays uniformly at
random over their n columns, and commit to the single PURE row that maximizes
your expected payoff under that assumption. This is exactly the checker's own
internal baseline B, so this solution reproduces Ratio ~= 0.1.
"""
import sys


def main():
    data = sys.stdin.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        v = data[pos]
        pos += 1
        return v

    m = int(nxt())
    n = int(nxt())
    A = [[int(nxt()) for _ in range(n)] for _ in range(m)]
    # historical log (N, H) is intentionally not read: this tier ignores it.

    col_avg = [sum(A[i][j] for j in range(n)) / n for i in range(m)]
    i_star = 0
    for i in range(1, m):
        if col_avg[i] > col_avg[i_star]:
            i_star = i

    p = [0.0] * m
    p[i_star] = 1.0
    print(" ".join(f"{x:.9f}" for x in p))


if __name__ == "__main__":
    main()
