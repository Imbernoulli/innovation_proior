# TIER: greedy
import sys

# Textbook "zero-skew" clock buffer insertion: fully equalize every net's
# delay to the slowest net, using whichever buffer type closes each gap in
# the FEWEST extra buffers (also the cheapest type per unit of delay here).
# This ignores process-variation exposure entirely.


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
        Var = [int(nxt()) for _ in range(C)]
        types.append((D, P, Var))
    nxt(); nxt()  # budgets -- greedy does not read them; it always fully balances
    ws = [int(nxt()) for _ in range(K)]

    best_t = max(range(m), key=lambda t: types[t][0])  # fewest insertions
    Db = types[best_t][0]
    Tstar = max(ws)

    counts = [[0] * m for _ in range(K)]
    for i in range(K):
        gap = Tstar - ws[i]
        if gap > 0:
            counts[i][best_t] = -(-gap // Db)

    out = []
    for row in counts:
        out.append(" ".join(map(str, row)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
