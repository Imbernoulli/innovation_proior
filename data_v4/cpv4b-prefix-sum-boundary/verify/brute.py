import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    w = int(data[idx]); idx += 1
    L = int(data[idx]); idx += 1
    R = int(data[idx]); idx += 1
    a = []
    for _ in range(n):
        a.append(int(data[idx])); idx += 1

    count = 0
    # Independent brute force: explicitly enumerate every block of exactly w
    # consecutive parcels by slicing, sum it directly (no prefix sums), and
    # test inclusive membership in [L, R].
    if 1 <= w <= n:
        for s in range(0, n - w + 1):          # 0-indexed start, s+w-1 <= n-1
            block = a[s:s + w]                  # exactly w elements
            total = sum(block)
            if L <= total <= R:
                count += 1
    print(count)

main()
