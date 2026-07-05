# TIER: trivial
"""Schoolbook split: form every scalar product x_i * y_j (p*q of them) and mix
them linearly into each output.  This reproduces the checker's baseline (R = p*q)
-> Ratio ~ 0.1.  Term for (i,j): a = e_i, c = e_j, d = T[:,i,j]."""
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    p = int(next(it)); q = int(next(it)); r = int(next(it))
    T = [[[0] * q for _ in range(p)] for _ in range(r)]
    for k in range(r):
        for i in range(p):
            for j in range(q):
                T[k][i][j] = int(next(it))

    lines = []
    R = 0
    for i in range(p):
        for j in range(q):
            a = [0] * p; a[i] = 1
            c = [0] * q; c[j] = 1
            d = [T[k][i][j] for k in range(r)]
            lines.append(" ".join(map(str, a + c + d)))
            R += 1
    sys.stdout.write(str(R) + "\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
