import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        # empty input -> n = 0
        print(0)
        return
    idx = 0
    n = int(data[idx]); idx += 1
    a = []
    for _ in range(n):
        a.append(int(data[idx])); idx += 1
    # Count subarrays [l, r], 0 <= l <= r < n, with sum(a[l..r]) < 0.
    # Independent O(n^2) brute force over all subarrays.
    count = 0
    for l in range(n):
        s = 0
        for r in range(l, n):
            s += a[r]
            if s < 0:
                count += 1
    print(count)

main()
