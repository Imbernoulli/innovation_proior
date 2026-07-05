# TIER: trivial
# First non-singular seeded random +/-1 completion respecting the pre-wired cells.
# |det| lands near the low end of the random distribution -> Ratio ~= 0.1.
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
    rng = random.Random(20260701 + n)
    M = completion(n, fixed, rng)
    for _ in range(200):
        if bareiss_det(M) != 0:
            break
        M = completion(n, fixed, rng)
    sys.stdout.write("\n".join(" ".join(map(str, row)) for row in M) + "\n")

if __name__ == "__main__":
    main()
