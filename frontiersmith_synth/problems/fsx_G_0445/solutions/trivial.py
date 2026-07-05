# TIER: trivial
import sys


def main():
    tok = sys.stdin.read().split()
    idx = 0
    n1 = int(tok[idx]); n2 = int(tok[idx + 1]); n3 = int(tok[idx + 2]); idx += 3
    T = [[[0] * n3 for _ in range(n2)] for _ in range(n1)]
    for k in range(n3):
        for i in range(n1):
            for j in range(n2):
                T[i][j][k] = int(tok[idx]); idx += 1

    # One product a_i*b_j per nonzero (i,j) support cell (== checker baseline).
    prods = []
    for i in range(n1):
        for j in range(n2):
            if any(T[i][j][k] != 0 for k in range(n3)):
                u = [0] * n1; u[i] = 1
                v = [0] * n2; v[j] = 1
                w = [T[i][j][k] for k in range(n3)]
                prods.append((u, v, w))

    lines = [str(len(prods))]
    for u, v, w in prods:
        lines.append(" ".join(map(str, u + v + w)))
    sys.stdout.write("\n".join(lines) + "\n")


main()
