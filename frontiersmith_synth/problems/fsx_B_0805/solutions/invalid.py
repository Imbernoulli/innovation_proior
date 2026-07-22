# TIER: invalid
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    M = int(next(it))
    # Emit an infeasible artifact: every price set far above its allowed
    # markup ceiling (violates cost_j <= p_j <= cost_j+markup_j).
    K = int(next(it))
    costs = []
    markups = []
    for _ in range(M):
        c = int(next(it)); mk = int(next(it))
        costs.append(c); markups.append(mk)
    prices = [costs[j] + markups[j] + 10_000 for j in range(M)]
    print(*prices)


if __name__ == "__main__":
    main()
