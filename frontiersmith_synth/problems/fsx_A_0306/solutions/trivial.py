# TIER: trivial
# Baseline: the triangular reference completion (matches the checker's internal baseline).
import sys

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    N = int(next(it)); K = int(next(it))
    fixed = {}
    for _ in range(K):
        r = int(next(it)); c = int(next(it)); v = int(next(it))
        fixed[(r, c)] = v
    rows = []
    for i in range(N):
        rows.append(" ".join(str(fixed.get((i, j), 1 if i >= j else -1)) for j in range(N)))
    sys.stdout.write("\n".join(rows) + "\n")

main()
