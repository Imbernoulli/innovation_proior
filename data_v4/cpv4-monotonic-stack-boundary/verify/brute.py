import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    a = [int(data[idx + i]) for i in range(n)]
    idx += n
    MOD = 1000000007
    total = 0
    # Sum of the minimum over every contiguous subarray a[l..r].
    for l in range(n):
        cur = None
        for r in range(l, n):
            if cur is None or a[r] < cur:
                cur = a[r]
            total += cur
    print(total % MOD)

if __name__ == "__main__":
    main()
