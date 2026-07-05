# TIER: trivial
"""Emit exactly the checker's internal baseline scheme -> scores ~0.1.

The baseline just fills every non-fixed inter-station cell from a fixed LCG
stream (the "default wiring"), with no attempt to raise the determinant.
"""
import sys


def bareiss_det(M):
    n = len(M)
    M = [row[:] for row in M]
    sign = 1
    prev = 1
    for k in range(n - 1):
        if M[k][k] == 0:
            piv = -1
            for i in range(k + 1, n):
                if M[i][k] != 0:
                    piv = i
                    break
            if piv < 0:
                return 0
            M[k], M[piv] = M[piv], M[k]
            sign = -sign
        mkk = M[k][k]
        Mk = M[k]
        for i in range(k + 1, n):
            Mi = M[i]
            mik = Mi[k]
            for j in range(k + 1, n):
                Mi[j] = (Mi[j] * mkk - mik * Mk[j]) // prev
        prev = mkk
    return sign * M[n - 1][n - 1]


class LCG:
    __slots__ = ("x",)

    def __init__(self, seed):
        self.x = (seed * 2862933555777941757 + 3037000493) & ((1 << 64) - 1)

    def nxt(self):
        self.x = (self.x * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return self.x

    def bit(self):
        return 1 if (self.nxt() >> 33) & 1 else -1


def build_baseline(n, seed, fixed):
    for attempt in range(64):
        rng = LCG(seed + attempt * 7919)
        M = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if (i, j) in fixed:
                    M[i][j] = fixed[(i, j)]
                else:
                    M[i][j] = rng.bit()
        if bareiss_det(M) != 0:
            return M
    return M


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    n = int(next(it)); seed = int(next(it)); k = int(next(it))
    fixed = {}
    for _ in range(k):
        i = int(next(it)); j = int(next(it)); v = int(next(it))
        fixed[(i, j)] = v
    M = build_baseline(n, seed, fixed)
    sys.stdout.write("\n".join(" ".join(str(x) for x in row) for row in M) + "\n")


if __name__ == "__main__":
    main()
