# TIER: trivial
# Reproduce the checker's internal baseline exactly: seeded pseudo-random +/-1
# matrix with the fixed first row, lightly improved by a deterministic greedy
# descent. This lands on the normalizer B -> Ratio ~ 0.1.
import sys, random

def gram(M, n):
    G = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0
            for k in range(n):
                s += M[k][i] * M[k][j]
            G[i][j] = s
    return G

def descend(M, G, n, rng, iters):
    for _ in range(iters):
        r = rng.randrange(1, n)
        c = rng.randrange(n)
        v = M[r][c]
        dE = 0
        for j in range(n):
            if j == c:
                continue
            new = G[c][j] - 2 * v * M[r][j]
            dE += new * new - G[c][j] * G[c][j]
        if dE <= 0:
            for j in range(n):
                if j == c:
                    continue
                new = G[c][j] - 2 * v * M[r][j]
                G[c][j] = new
                G[j][c] = new
            M[r][c] = -v

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    n = int(next(it))
    r0 = [int(next(it)) for _ in range(n)]
    rng = random.Random(90000 + n)
    M = [list(r0)] + [[rng.choice((-1, 1)) for _ in range(n)] for _ in range(n - 1)]
    G = gram(M, n)
    descend(M, G, n, rng, 3 * n)
    sys.stdout.write("\n".join(" ".join(str(x) for x in row) for row in M) + "\n")

main()
