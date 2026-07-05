# TIER: trivial
# One scalar multiplier per nonzero crosstalk coefficient: the naive per-entry
# CP decomposition.  Stage for coefficient T[i][j][k] = e_i*value (x) e_j (x) e_k.
# R = number of nonzero entries -> reproduces the checker baseline (Ratio ~ 0.1).
import sys
from fractions import Fraction


def main():
    tok = sys.stdin.read().split()
    it = iter(tok)
    a = int(next(it)); b = int(next(it)); c = int(next(it))
    T = [[[int(next(it)) for _ in range(c)] for _ in range(b)] for _ in range(a)]

    stages = []
    for i in range(a):
        for j in range(b):
            for k in range(c):
                val = T[i][j][k]
                if val == 0:
                    continue
                u = [0] * a; u[i] = val
                v = [0] * b; v[j] = 1
                w = [0] * c; w[k] = 1
                stages.append((u, v, w))

    lines = [str(len(stages))]
    for (u, v, w) in stages:
        lines.append(" ".join(str(x) for x in u + v + w))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
