# TIER: strong
# Restart + sign-flip hill climbing. Seed several random +/-1 grids, take the best
# by exact |det|, then greedily flip single signal cells whenever a flip increases
# |det| (a few sweeps). Finally renormalize column 0 to all +1. Deterministic:
# seed depends only on N. Because N is odd, the Hadamard bound is unreachable, so
# this stays below saturation but consistently beats plain random search.
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
    rng = random.Random(999 + n)
    best = None
    best_d = -1
    for _ in range(20):
        M = [[rng.choice((-1, 1)) for _ in range(n)] for _ in range(n)]
        d = abs(bareiss_det(M))
        if d > best_d:
            best_d = d
            best = [r[:] for r in M]

    cur = best
    cur_d = best_d
    for _ in range(2):
        improved = False
        for i in range(n):
            for j in range(n):
                cur[i][j] = -cur[i][j]
                nd = abs(bareiss_det(cur))
                if nd > cur_d:
                    cur_d = nd
                    improved = True
                else:
                    cur[i][j] = -cur[i][j]
        if not improved:
            break

    cur = normalize(cur)
    out = "\n".join(" ".join(map(str, row)) for row in cur)
    sys.stdout.write(out + "\n")

if __name__ == "__main__":
    main()
