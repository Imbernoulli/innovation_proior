import sys
from itertools import product

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it)); m = int(next(it))
    p = [[int(next(it)) for _ in range(m)] for _ in range(n)]

    # Brute force: each parcel i independently chooses a "decision":
    #   -1  -> undelivered
    #    j  -> delivered into slot j (0..m-1)
    # Constraint: no two parcels share a slot. Maximize total profit.
    # We allow the empty assignment (all -1), so answer >= 0.
    best = 0  # empty selection
    choices = [list(range(-1, m)) for _ in range(n)]
    for assign in product(*choices):
        used = set()
        ok = True
        total = 0
        for i, j in enumerate(assign):
            if j == -1:
                continue
            if j in used:
                ok = False
                break
            used.add(j)
            total += p[i][j]
        if ok:
            best = max(best, total)
    print(best)

main()
