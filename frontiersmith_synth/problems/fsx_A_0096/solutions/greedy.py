# TIER: greedy
# Start from the baseline layout and run a modest single-chip greedy descent:
# repeatedly flip a chip (in rows 1..n-1) whenever it does not increase the total
# off-diagonal Gram energy. A limited budget -> beats the baseline but does not
# reach the deeper optima that multi-restart search finds.
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
    # rebuild the baseline (same construction as trivial), then improve further
    rng = random.Random(90000 + n)
    M = [list(r0)] + [[rng.choice((-1, 1)) for _ in range(n)] for _ in range(n - 1)]
    G = gram(M, n)
    descend(M, G, n, rng, 3 * n)          # baseline warm-up
    descend(M, G, n, random.Random(500 + n), 120)  # modest extra descent
    sys.stdout.write("\n".join(" ".join(str(x) for x in row) for row in M) + "\n")

main()
