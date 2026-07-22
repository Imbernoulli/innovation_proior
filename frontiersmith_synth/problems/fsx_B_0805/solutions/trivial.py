# TIER: trivial
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    M = int(next(it))
    K = int(next(it))
    costs = []
    markups = []
    for _ in range(M):
        costs.append(int(next(it)))
        markups.append(int(next(it)))
    # (cohort data is unused by the trivial construction)

    prices = [costs[j] + max(1, markups[j] // 5) for j in range(M)]
    print(*prices)


if __name__ == "__main__":
    main()
