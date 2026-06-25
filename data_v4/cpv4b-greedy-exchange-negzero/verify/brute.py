import sys
from itertools import combinations


def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    L = int(data[idx]); idx += 1
    K = int(data[idx]); idx += 1
    c = [int(data[idx + i]) for i in range(n)]
    idx += n

    # Independent brute force: enumerate EVERY subset whose size t satisfies
    # L <= t <= K, take its sum, keep the maximum. No greedy assumptions.
    best = None
    for t in range(L, K + 1):
        for combo in combinations(range(n), t):
            s = sum(c[i] for i in combo)
            if best is None or s > best:
                best = s

    # If L == 0 the empty subset (size 0, sum 0) is always among the candidates,
    # so `best` is well defined whenever 0 <= L <= K <= n.
    print(best)


if __name__ == "__main__":
    main()
