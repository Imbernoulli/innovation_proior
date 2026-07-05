# TIER: greedy
"""Structured circulant schedule: row i = beacon pattern r cyclically shifted by i.
Row 0 is r (shift 0) so the beacon constraint holds automatically. Circulant +/-1
matrices carry substantially more independence than the staircase baseline. If the
circulant happens to be singular, flip one interior entry to restore full rank."""
import sys


def bareiss_det(M):
    n = len(M)
    A = [row[:] for row in M]
    sign = 1
    prev = 1
    for k in range(n - 1):
        if A[k][k] == 0:
            sw = None
            for i in range(k + 1, n):
                if A[i][k] != 0:
                    sw = i
                    break
            if sw is None:
                return 0
            A[k], A[sw] = A[sw], A[k]
            sign = -sign
        akk = A[k][k]
        for i in range(k + 1, n):
            Ai = A[i]
            aik = Ai[k]
            Ak = A[k]
            for j in range(k + 1, n):
                Ai[j] = (Ai[j] * akk - aik * Ak[j]) // prev
        prev = akk
    return sign * A[n - 1][n - 1]


def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    r = [int(x) for x in data[1:1 + N]]
    # circulant: M[i][j] = r[(j - i) mod N]; row 0 = r
    M = [[r[(j - i) % N] for j in range(N)] for i in range(N)]
    if bareiss_det(M) == 0:
        # restore rank with a single interior flip (never touches row 0)
        M[1][0] = -M[1][0]
        if bareiss_det(M) == 0:
            M[N - 1][N - 1] = -M[N - 1][N - 1]
    out = [" ".join(str(x) for x in M[i]) for i in range(N)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
