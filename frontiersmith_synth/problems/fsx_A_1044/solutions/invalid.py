# TIER: invalid
import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); p = int(next(it))
    # Deliberately infeasible: every cell seeded (n nonzeros, blows any sparsity budget
    # s < n) AND the values sit outside the valid range [0, p-1].
    x0 = [p + j + 1 for j in range(n)]
    print(" ".join(map(str, x0)))

main()
