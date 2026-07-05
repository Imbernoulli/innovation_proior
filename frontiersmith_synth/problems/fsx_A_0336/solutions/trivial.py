# TIER: trivial
"""Staircase baseline: lower-triangular +1 (on/below diag), -1 above diag, then flip
whole columns so row 0 matches the beacon r. |det| = 2^{N-1} (column flips only change
the sign), so this reproduces the checker baseline B = N-1  ->  Ratio ~ 0.1."""
import sys


def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    r = [int(x) for x in data[1:1 + N]]
    M = [[1 if j <= i else -1 for j in range(N)] for i in range(N)]
    # flip columns so that row 0 equals r (does not change |det|)
    for j in range(N):
        if M[0][j] != r[j]:
            for i in range(N):
                M[i][j] = -M[i][j]
    out = []
    for i in range(N):
        out.append(" ".join(str(x) for x in M[i]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
