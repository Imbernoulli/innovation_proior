# TIER: trivial
# Naive scheme: compute every product x_i*y_j (m*n multiplications), then combine
# linearly per output. R = m*n -> reproduces the checker baseline -> ratio ~ 0.1.
import sys

def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    P = int(next(it)); m = int(next(it)); n = int(next(it))
    T = [[[int(next(it)) for _ in range(n)] for _ in range(m)] for _ in range(P)]

    out = []
    R = m * n
    out.append(str(R))
    for i in range(m):
        for j in range(n):
            a = ["1" if k == i else "0" for k in range(m)]
            b = ["1" if k == j else "0" for k in range(n)]
            c = [str(T[p][i][j]) for p in range(P)]
            out.append(" ".join(a)); out.append(" ".join(b)); out.append(" ".join(c))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
