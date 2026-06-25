import sys
from functools import lru_cache

def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    K = int(data[idx]); idx += 1
    L = int(data[idx]); idx += 1
    a = []
    for _ in range(n):
        a.append(int(data[idx])); idx += 1

    # Brute force: try EVERY partition of panels 1..n into contiguous blocks,
    # each block of inclusive length 1..L. Cost of a block is K + max over it.
    # solve(i) = min total cost to tile panels i..n (0-indexed start i).
    sys.setrecursionlimit(10000)

    @lru_cache(maxsize=None)
    def solve(i):
        if i == n:
            return 0
        best = None
        cur_max = None
        # block covers a[i..j] inclusive, length j-i+1 in 1..L
        for j in range(i, min(n, i + L)):
            cur_max = a[j] if cur_max is None else max(cur_max, a[j])
            total = K + cur_max + solve(j + 1)
            if best is None or total < best:
                best = total
        return best

    print(solve(0))

if __name__ == "__main__":
    main()
