# TIER: trivial
# Reproduce the checker's baseline block grid -> scores ~0.1.
import sys


def build_baseline(N):
    M = [[0] * N for _ in range(N)]
    pos = 0
    blk = [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
    while pos + 3 <= N:
        for a in range(3):
            for b in range(3):
                M[pos + a][pos + b] = blk[a][b]
        pos += 3
    r = N - pos
    if r == 1:
        M[pos][pos] = 1
    elif r == 2:
        M[pos][pos] = 1
        M[pos][pos + 1] = 1
        M[pos + 1][pos] = 1
        M[pos + 1][pos + 1] = 0
    return M


def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    M = build_baseline(N)
    out = []
    for i in range(N):
        out.append(" ".join(str(x) for x in M[i]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
