# TIER: greedy
# Seeded multi-restart random sampling: draw many +/-1 completions that respect the
# pre-wired cells and keep the one with the largest exact |det|. No local search, so
# it sits between the trivial completion and the hill-climbed strong array.
import sys, random

def bareiss_det(M):
    n = len(M); M = [r[:] for r in M]; sign = 1; prev = 1
    for k in range(n - 1):
        if M[k][k] == 0:
            sw = None
            for r in range(k + 1, n):
                if M[r][k] != 0:
                    sw = r; break
            if sw is None:
                return 0
            M[k], M[sw] = M[sw], M[k]; sign = -sign
        for r in range(k + 1, n):
            for c in range(k + 1, n):
                M[r][c] = (M[r][c] * M[k][k] - M[r][k] * M[k][c]) // prev
        prev = M[k][k]
    return sign * M[n - 1][n - 1]

def parse():
    tok = sys.stdin.read().split()
    n = int(tok[0]); nf = int(tok[1]); fixed = {}
    idx = 2
    for _ in range(nf):
        r = int(tok[idx]); c = int(tok[idx + 1]); v = int(tok[idx + 2]); idx += 3
        fixed[(r, c)] = v
    return n, fixed

def completion(n, fixed, rng):
    M = [[rng.choice((-1, 1)) for _ in range(n)] for _ in range(n)]
    for (i, j), v in fixed.items():
        M[i][j] = v
    return M

def main():
    n, fixed = parse()
    rng = random.Random(31337 + n)
    best = None; bd = -1
    for _ in range(40):
        M = completion(n, fixed, rng)
        d = abs(bareiss_det(M))
        if d > bd:
            bd = d; best = M
    sys.stdout.write("\n".join(" ".join(map(str, row)) for row in best) + "\n")

if __name__ == "__main__":
    main()
