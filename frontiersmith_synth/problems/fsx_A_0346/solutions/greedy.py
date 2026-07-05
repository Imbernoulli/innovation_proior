# TIER: greedy
# Seeded multi-restart random search: draw many normalized +/-1 grids and keep the
# one with the largest exact |det|. Each grid is normalized so column 0 is all +1
# (flip whole rows -- preserves |det|). Deterministic: seed depends only on N.
# Beats the corridor baseline but stays well below what local search finds.
import sys, random

def bareiss_det(M):
    n = len(M)
    M = [row[:] for row in M]
    sign = 1
    prev = 1
    for k in range(n - 1):
        if M[k][k] == 0:
            sw = None
            for r in range(k + 1, n):
                if M[r][k] != 0:
                    sw = r
                    break
            if sw is None:
                return 0
            M[k], M[sw] = M[sw], M[k]
            sign = -sign
        for r in range(k + 1, n):
            for c in range(k + 1, n):
                M[r][c] = (M[r][c] * M[k][k] - M[r][k] * M[k][c]) // prev
        prev = M[k][k]
    return sign * M[n - 1][n - 1]

def normalize(M):
    for row in M:
        if row[0] == -1:
            for j in range(len(row)):
                row[j] = -row[j]
    return M

def main():
    n = int(sys.stdin.read().split()[0])
    rng = random.Random(1234 + n)
    best = None
    best_d = -1
    for _ in range(60):
        M = [[rng.choice((-1, 1)) for _ in range(n)] for _ in range(n)]
        d = abs(bareiss_det(M))
        if d > best_d:
            best_d = d
            best = M
    best = normalize(best)
    out = "\n".join(" ".join(map(str, row)) for row in best)
    sys.stdout.write(out + "\n")

if __name__ == "__main__":
    main()
