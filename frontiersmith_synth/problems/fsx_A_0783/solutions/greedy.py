# TIER: greedy
"""The obvious recipe: doses matter most where the most people live, so fully protect
the highest-population cities first until the budget runs out. This ignores network
position entirely -- it is exactly the per-city-importance trap the problem plants:
it never notices that a handful of cheap, low-population bridge cities separate
several scenarios' spread trees at once."""
import sys, math

def ceil_cost(pop, alpha_percent):
    return -(-pop * alpha_percent // 100)

def main():
    it = iter(sys.stdin.read().split())
    def nx():
        return next(it)
    N = int(nx()); K = int(nx()); T = int(nx())
    alpha_percent = int(nx()); budget = int(nx())
    pops = [int(nx()) for _ in range(N)]
    M = int(nx())
    for _ in range(M):
        nx(); nx(); nx()
    for _ in range(K):
        nx(); nx()

    costs = [ceil_cost(pops[i], alpha_percent) for i in range(N)]
    order = sorted(range(N), key=lambda i: -pops[i])
    doses = [0] * N
    remaining = budget
    for i in order:
        if remaining <= 0:
            break
        if costs[i] <= remaining:
            doses[i] = costs[i]
            remaining -= costs[i]
        else:
            doses[i] = remaining
            remaining = 0
    print(" ".join(str(d) for d in doses))

if __name__ == "__main__":
    main()
