# TIER: trivial
# One scalar multiplication per nonzero transfer entry: for each (i,j,k) with
# T[i][j][k] != 0 emit a stage that places that value at exactly that cell using
# axis-aligned indicator vectors.  R = number of nonzero entries = the checker
# baseline B, so this scores ~0.1.
import sys


def main():
    tok = sys.stdin.read().split()
    it = iter(tok)
    I = int(next(it)); J = int(next(it)); K = int(next(it))
    T = [[[int(next(it)) for _ in range(K)] for _ in range(J)] for _ in range(I)]

    stages = []
    for i in range(I):
        for j in range(J):
            for k in range(K):
                val = T[i][j][k]
                if val == 0:
                    continue
                u = [0] * I; v = [0] * J; w = [0] * K
                u[i] = val; v[j] = 1; w[k] = 1
                stages.append(u + v + w)

    outl = [str(len(stages))]
    for s in stages:
        outl.append(" ".join(str(x) for x in s))
    sys.stdout.write("\n".join(outl) + "\n")


if __name__ == "__main__":
    main()
