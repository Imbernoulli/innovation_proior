# TIER: strong
# Multi-restart hill-climbing. From many seeded random starts (row 0 pinned to
# r0) run single-chip greedy descent to (near) convergence using the O(n)
# incremental Gram update, and keep the lowest-energy layout found. Explores far
# more of the sign-matrix landscape than the single greedy descent, so it reaches
# markedly lower interference energy with a per-instance-varying result.
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

def energy_from_G(G, n):
    return sum(G[i][j] * G[i][j] for i in range(n) for j in range(i + 1, n))

def descend(M, G, n, rng, iters):
    E = energy_from_G(G, n)
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
            E += dE
    return E

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    n = int(next(it))
    r0 = [int(next(it)) for _ in range(n)]
    rng = random.Random(222 + n)
    best_M = None
    best_E = None
    for _ in range(10):
        M = [list(r0)] + [[rng.choice((-1, 1)) for _ in range(n)] for _ in range(n - 1)]
        G = gram(M, n)
        E = descend(M, G, n, rng, 4000)
        if best_E is None or E < best_E:
            best_E = E
            best_M = [row[:] for row in M]
    sys.stdout.write("\n".join(" ".join(str(x) for x in row) for row in best_M) + "\n")

main()
