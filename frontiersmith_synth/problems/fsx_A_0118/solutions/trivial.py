# TIER: trivial
# Naive one-multiplier-per-nonzero-coefficient decomposition: for every nonzero
# entry T[i][j][k] emit a stage  (val * e_i) (x) e_j (x) e_k.  R = nnz = the
# checker's baseline B, so this reproduces the baseline exactly (Ratio ~ 0.1).
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    a = int(next(it)); b = int(next(it)); c = int(next(it))
    T = [[[0] * c for _ in range(b)] for _ in range(a)]
    for i in range(a):
        for j in range(b):
            for k in range(c):
                T[i][j][k] = int(next(it))

    stages = []
    for i in range(a):
        for j in range(b):
            for k in range(c):
                if T[i][j][k] != 0:
                    u = [0] * a; u[i] = T[i][j][k]
                    v = [0] * b; v[j] = 1
                    w = [0] * c; w[k] = 1
                    stages.append((u, v, w))

    lines = [str(len(stages))]
    for u, v, w in stages:
        lines.append(" ".join(str(x) for x in (u + v + w)))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
