# TIER: trivial
import sys

# Reproduces the checker's own reference plan exactly: use only the "safe"
# buffer type (type 0, low variation) and fully close every net's gap to
# the slowest net. Safe, but power-wasteful.


def main():
    data = sys.stdin.read().split()
    p = 0
    def nxt():
        nonlocal p
        v = data[p]; p += 1; return v
    K = int(nxt()); m = int(nxt()); C = int(nxt())
    types = []
    for _ in range(m):
        D = int(nxt()); P = int(nxt())
        for _ in range(C):
            nxt()
        types.append((D, P))
    nxt(); nxt()  # budgets, unused
    ws = [int(nxt()) for _ in range(K)]

    D0 = types[0][0]
    Tstar = max(ws)
    counts = [[0] * m for _ in range(K)]
    for i in range(K):
        gap = Tstar - ws[i]
        if gap > 0:
            counts[i][0] = -(-gap // D0)

    out = []
    for row in counts:
        out.append(" ".join(map(str, row)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
