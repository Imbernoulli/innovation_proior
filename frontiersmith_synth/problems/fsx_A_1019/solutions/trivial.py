# TIER: trivial
import sys


def main():
    data = sys.stdin.read().split()
    idx = [0]

    def nxt():
        v = data[idx[0]]
        idx[0] += 1
        return v

    N = int(nxt()); G = int(nxt()); K = int(nxt()); R = int(nxt()); eps = int(nxt())
    cap = [int(nxt()) for _ in range(G)]
    S = [int(nxt()) for _ in range(G)]
    for _ in range(N):
        nxt()  # comm
        for _ in range(G):
            nxt()  # weight

    M = max(1, round(0.85 * N))
    x = [[0] * G for _ in range(N)]
    for g in range(G):
        base, rem = S[g] // M, S[g] % M
        for i in range(M):
            x[i][g] = base + (1 if i < rem else 0)

    print("\n".join(" ".join(map(str, row)) for row in x))


if __name__ == "__main__":
    main()
