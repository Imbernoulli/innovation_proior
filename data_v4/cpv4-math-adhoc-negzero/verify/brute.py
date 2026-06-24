import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        # no input at all -> n = 0 -> only empty subarray, product 1
        print(1)
        return
    it = iter(data)
    n = int(next(it))
    a = [int(next(it)) for _ in range(n)]

    # Empty subarray is allowed; its product is the empty product = 1.
    best = 1
    # Try every contiguous subarray [i, j) with i < j (non-empty).
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= a[j]
            if prod > best:
                best = prod
    print(best)

main()
