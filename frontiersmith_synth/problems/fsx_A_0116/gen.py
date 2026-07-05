import sys, random

def bareiss_det(M):
    n = len(M)
    M = [row[:] for row in M]
    sign = 1
    prev = 1
    for k in range(n - 1):
        if M[k][k] == 0:
            sw = -1
            for i in range(k + 1, n):
                if M[i][k] != 0:
                    sw = i
                    break
            if sw == -1:
                return 0
            M[k], M[sw] = M[sw], M[k]
            sign = -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                M[i][j] = (M[i][j] * M[k][k] - M[i][k] * M[k][j]) // prev
        prev = M[k][k]
    return sign * M[n - 1][n - 1]

def default_fill(n, forced):
    # terrain-default corridor signs: downstream/self links +1, upstream links -1
    A = [[(1 if j >= i else -1) for j in range(n)] for i in range(n)]
    for (r, c), v in forced.items():
        A[r][c] = v
    return A

def main():
    idx = int(sys.argv[1])
    rng = random.Random(4116000 + idx)
    ladder = [4, 4, 5, 5, 5, 6, 6, 6, 7, 7]
    n = ladder[(idx - 1) % len(ladder)]
    frac = 0.12 + 0.06 * ((idx - 1) % 3)   # 0.12, 0.18, 0.24 cycling
    k = int(round(frac * n * n))
    k = max(2, k)
    positions = [(r, c) for r in range(n) for c in range(n)]
    # sample a set of terrain-fixed links whose default completion is non-singular
    while True:
        chosen = rng.sample(positions, k)
        forced = {p: rng.choice([-1, 1]) for p in chosen}
        if bareiss_det(default_fill(n, forced)) != 0:
            break
    lines = ["%d %d" % (n, k)]
    for (r, c) in sorted(forced):
        lines.append("%d %d %d" % (r, c, forced[(r, c)]))
    sys.stdout.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
